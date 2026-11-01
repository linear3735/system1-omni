"""H800 派发：运行合同跨进程传递的 CPU 回归（不连集群、不申请 GPU）。

这里的 producer 片段、sbatch 包装器与 env 模板都是仓库里的真文件；child 由真正的
exec 启动，父进程只给 PATH 与标为 fixture 的 SLURM_JOB_ID，不预先 export 任何
待测变量。所以它抓的是 bash -n 与纯文本检查都看不见的那一类失败：run.env 里的
普通赋值没有跨过 exec，作业在分配里立刻以 “RUN_DIR is required” 死掉。
"""

from __future__ import annotations

import re
import shlex
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

CLEAN_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
PRODUCER_END = re.compile(r'^\s*\} >"\$(\w+)/run\.env"$', re.MULTILINE)

# 两个仓库都必须写进 run.env 的运行合同。少一个键，child 就会退回自己的默认值。
CONTRACT_KEYS = (
    "RUN_DIR",
    "SRC_DIR",
    "OUT_DIR",
    "REVISION",
    "STAGES",
    "PYTHON",
    "CUDA_MODULE",
    "SHARED_TOOLS",
    "TEST_ARGS",
    "TEST_ENV_FILE",
    "JOB_NAME",
    "RUN_ID",
)
# 只有平台仓库的 producer 写这一个；缺失意味着测试阶段装不上依赖。
PLATFORM_ONLY_KEYS = ("TEST_REQUIREMENTS",)

# child 只做一件事：把 exec 之后真正收到的环境原样记下来。
CHILD = '#!/usr/bin/env bash\nset -u\nenv | LC_ALL=C sort >"$OUT_DIR/child-env.txt"\n'

FIXTURES = {
    "revision": "0123456789abcdef",
    "stages": "env,probe",
    "python_bin": "python3",
    "cuda_module": "cuda/13.0",
    "shared_tools": "",
    "test_args": "-p omni-jev --test cuda -- --ignored",
    "job_name": "dsv41-quant-h800",
    "run_id": "20261101T120000Z-0123456",
    "test_requirements": "pytest==9.2.0 pyyaml==6.0.3",
}


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools/h800/jobs/run_h800.sbatch").is_file():
            return candidate
    raise RuntimeError("找不到仓库根：tools/h800/jobs/run_h800.sbatch 不在任何父目录下")


def bash(script: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    """在干净环境里跑一段 bash：只留 PATH 与显式 fixture。"""
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": CLEAN_PATH, **extra_env},
        check=False,
    )


def producer_fragment(dispatch: Path) -> tuple[str, str]:
    """取出 dispatch.sh 里真正写 run.env 的那一段，以及它用的临时目录变量名。"""
    text = dispatch.read_text(encoding="utf-8")
    matches = list(PRODUCER_END.finditer(text))
    if not matches:
        raise AssertionError("dispatch.sh 里找不到写 run.env 的片段")
    match = matches[-1]
    opening = text.rindex("\n    {\n", 0, match.start())
    return text[opening + 1 : match.end()], match.group(1)


def produced_run_env(tmp: Path, run_dir: Path) -> str:
    """用真 producer 片段生成 run.env，不手抄键名。"""
    fragment, tmp_var = producer_fragment(repo_root() / "tools/h800/dispatch.sh")
    assignments = [f"{tmp_var}={shlex.quote(str(tmp))}", f"run_dir={shlex.quote(str(run_dir))}"]
    assignments += [f"{key}={shlex.quote(value)}" for key, value in FIXTURES.items()]
    result = bash("set -euo pipefail\n" + "\n".join(assignments) + "\n" + fragment)
    if result.returncode != 0:
        raise AssertionError(f"producer 片段执行失败：{result.stderr}")
    return (tmp / "run.env").read_text(encoding="utf-8")


def parse_env_dump(text: str) -> dict[str, str]:
    pairs = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            pairs[key] = value
    return pairs


def sourced_values(run_env: Path) -> dict[str, str]:
    """按 sbatch 的读法（set -a + source）取期望值，不手抄任何键名。"""
    result = bash(f"set -a\n. {shlex.quote(str(run_env))}\nset +a\nenv | LC_ALL=C sort\n")
    if result.returncode != 0:
        raise AssertionError(f"run.env 无法 source：{result.stderr}")
    return parse_env_dump(result.stdout)


class RunEnvContractTest(unittest.TestCase):
    """运行合同必须跨过 sbatch 的 exec 到达 child。"""

    def test_contract_survives_exec(self) -> None:
        sbatch = repo_root() / "tools/h800/jobs/run_h800.sbatch"
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            run_dir = tmp / "run"
            run_dir.mkdir()
            run_env = run_dir / "run.env"
            run_env.write_text(produced_run_env(tmp, run_dir), encoding="utf-8")
            expected = sourced_values(run_env)
            produced = set(expected) & set(re.findall(r"^([A-Z_]+)=", run_env.read_text(), re.M))

            # 真 sbatch 被复制进 run 目录再执行：dispatch.sh 就是这么发的。
            sbatch_copy = run_dir / "run_h800.sbatch"
            sbatch_copy.write_bytes(sbatch.read_bytes())

            entry = tmp / "child.sh"
            entry.write_text(CHILD, encoding="utf-8")
            entry.chmod(0o755)
            with tarfile.open(run_dir / "src.tar.gz", "w:gz") as archive:
                archive.add(entry, arcname="tools/h800/jobs/run_h800.sh")

            result = bash(
                f"cd {shlex.quote(str(run_dir))} && exec bash {shlex.quote(str(sbatch_copy))}",
                SLURM_JOB_ID="999999",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            dump = run_dir / "out" / "child-env.txt"
            self.assertTrue(dump.is_file(), "child 没有留下环境快照")
            received = parse_env_dump(dump.read_text(encoding="utf-8"))

        # 1. producer 写的每个键都必须跨过 exec，且值与 source run.env 的完全一致。
        self.assertTrue(set(CONTRACT_KEYS) <= produced, f"producer 漏了合同键：{produced}")
        if (repo_root() / "pyproject.toml").is_file():
            self.assertTrue(set(PLATFORM_ONLY_KEYS) <= produced, "producer 漏了 TEST_REQUIREMENTS")
        for key in sorted(produced):
            with self.subTest(key=key):
                self.assertIn(key, received, f"{key} 没有跨过 exec")
                self.assertEqual(received[key], expected[key], f"{key} 的值在途中变了")
        # 2. 值本身必须就是我们塞进去的那些，避免 producer 与测试一起错。
        self.assertEqual(received["STAGES"], "env,probe", "阶段被静默改回默认值")
        self.assertEqual(received["RUN_DIR"], str(run_dir))
        self.assertEqual(received["SRC_DIR"], str(run_dir / "src"))
        self.assertEqual(received["OUT_DIR"], str(run_dir / "out"))
        self.assertEqual(received["REVISION"], FIXTURES["revision"])
        self.assertEqual(received["PYTHON"], FIXTURES["python_bin"])
        self.assertEqual(received["CUDA_MODULE"], FIXTURES["cuda_module"])
        self.assertEqual(received["SHARED_TOOLS"], "", "空值必须原样传成空值")
        self.assertEqual(received["TEST_ARGS"], FIXTURES["test_args"], "带空格的值被拆开了")
        self.assertEqual(received["TEST_ENV_FILE"], str(run_dir / "gpu-test.env"))
        self.assertEqual(received["JOB_NAME"], f"{FIXTURES['job_name']}-{FIXTURES['run_id']}")
        self.assertEqual(received["RUN_ID"], FIXTURES["run_id"])
        if "TEST_REQUIREMENTS" in produced:
            self.assertEqual(received["TEST_REQUIREMENTS"], FIXTURES["test_requirements"])

    def test_entry_still_demands_run_dir(self) -> None:
        """红侧：没有 sbatch 的导出，真入口必须在第一步就拒绝，而不是继续跑。"""
        entry = repo_root() / "tools/h800/jobs/run_h800.sh"
        result = bash(f"exec bash {shlex.quote(str(entry))}")
        self.assertNotEqual(result.returncode, 0, "缺 RUN_DIR 时入口竟然成功了")
        self.assertIn("RUN_DIR is required", result.stderr)


class EnvTemplateTest(unittest.TestCase):
    """模板里每个字面值都要原样取出：未加引号的空格值会静默截断。"""

    def test_template_values_round_trip(self) -> None:
        template = repo_root() / "tools/h800/h800.env.example"
        literal: dict[str, str] = {}
        for line in template.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            if quote := value[:1]:
                if quote in {"'", '"'} and value.endswith(quote) and len(value) > 1:
                    value = value[1:-1]
            literal[key] = value
        if (repo_root() / "pyproject.toml").is_file():
            self.assertIn("H800_TEST_REQUIREMENTS", literal, "模板里必须写明固定依赖")

        script = (
            f"set -a\n. {shlex.quote(str(template))}\nset +a\n"
            + "\n".join(f'printf "%s\\n" "${{{key}-}}"' for key in literal)
            + "\nenv | LC_ALL=C sort\n"
        )
        result = bash(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        values, env_dump = lines[: len(literal)], lines[len(literal) :]
        for key, value in zip(literal, values, strict=True):
            with self.subTest(key=key):
                self.assertEqual(value, literal[key], f"{key} 的字面值没有原样取出")
        sourced = parse_env_dump("\n".join(env_dump))
        for key in literal:
            self.assertIn(key, sourced, f"{key} 没有被 source 出来")
        for stray in ("pyyaml", "pytest"):
            self.assertNotIn(stray, sourced, f"模板里未加引号的空格值泄漏成了变量 {stray}")
