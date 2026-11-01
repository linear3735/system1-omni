"""H800 派发：等待循环与回传限额的 CPU 回归（不连集群、不申请 GPU）。

ssh 被换成 PATH 上的 stub，只替代传输与调度器；dispatch.sh 本身是真文件、真执行。
覆盖三种情形：成功但空的 squeue（作业已离队）、记账迟到、查询失败（未知，不是完成），
以及大小核查与打包必须用同一个文件集合、按 KiB 比较。
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

CLEAN_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
RUN_ID = "20261101T000000Z-stub"
ROOT = "/aifs4su/shiheming/ci-stub"

SSH_STUB = r"""#!/usr/bin/env bash
# 只替代传输与调度器：按命令子串给出脚本化响应，dispatch.sh 与 sbatch 保持真代码。
set -u
dir=${STUB_DIR:?}
cmd="$*"
log() { printf '%s\t%s\n' "$1" "$cmd" >>"$dir/calls.log"; }
next_line() {
    local file="$dir/$1" line
    if [ ! -s "$file" ]; then printf '\n'; return 0; fi
    line=$(head -n 1 "$file")
    if [ "$(wc -l <"$file")" -gt 1 ]; then
        tail -n +2 "$file" >"$file.next" && mv "$file.next" "$file"
    else
        : >"$file"
    fi
    printf '%s' "$line"
}
case "$cmd" in
    *scancel*) log scancel; exit 0 ;;
    *"sbatch --parsable"*) log sbatch; printf '424242\n'; exit 0 ;;
    *squeue*) log squeue; line=$(next_line squeue.seq); [ "$line" = FAIL ] && exit 255
              printf '%s\n' "$line"; exit 0 ;;
    *sacct*) log sacct; line=$(next_line sacct.seq); [ "$line" = FAIL ] && exit 255
             printf '%s\n' "$line"; exit 0 ;;
    *"du -sk"*) log du; printf '%s\n' "${STUB_EVIDENCE_KIB:-4}"; exit 0 ;;
    *"--ignore-failed-read"*) log fetch; tar -czf - -C "$dir/evidence" .; exit 0 ;;
    *"tar -xzf"*) log unpack; exit 0 ;;
    *"cat >"*) log upload; cat >/dev/null; exit 0 ;;
    *mkdir*) log mkdir; exit 0 ;;
    *) log other; exit 0 ;;
esac
"""


def repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools/h800/dispatch.sh").is_file():
            return candidate
    raise RuntimeError("找不到仓库根：tools/h800/dispatch.sh 不在任何父目录下")


class DispatchStubTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.stub = self.tmp / "stub"
        (self.stub / "bin").mkdir(parents=True)
        ssh = self.stub / "bin" / "ssh"
        ssh.write_text(SSH_STUB, encoding="utf-8")
        ssh.chmod(0o755)
        evidence = self.stub / "evidence" / "out"
        evidence.mkdir(parents=True)
        (evidence / "exit-code").write_text("0\n", encoding="utf-8")
        (evidence / "stage-status.tsv").write_text("env\tok\nprobe\tok\n", encoding="utf-8")
        self.artifacts = self.tmp / "artifacts"

    def dispatch(self, *, squeue: str, sacct: str, evidence_kib: int = 4, cap_mb: int = 1):
        (self.stub / "squeue.seq").write_text(squeue, encoding="utf-8")
        (self.stub / "sacct.seq").write_text(sacct, encoding="utf-8")
        env = {
            "PATH": f"{self.stub / 'bin'}:{CLEAN_PATH}",
            "HOME": str(self.tmp),
            "STUB_DIR": str(self.stub),
            "STUB_EVIDENCE_KIB": str(evidence_kib),
            "H800_HOST": "stub-host",
            "H800_ROOT": ROOT,
            "H800_ARTIFACT_DIR": str(self.artifacts),
            "H800_RUN_ID": RUN_ID,
            "H800_STAGES": "env,probe",
            "H800_WAIT_SECONDS": "30",
            "H800_ALLOW_DIRTY": "1",
            "H800_EVIDENCE_MAX_MB": str(cap_mb),
        }
        return subprocess.run(
            ["bash", str(repo_root() / "tools/h800/dispatch.sh"), "--submit"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=180,
        )

    def calls(self) -> list[tuple[str, str]]:
        log = self.stub / "calls.log"
        if not log.exists():
            return []
        parsed = []
        for line in log.read_text(encoding="utf-8").splitlines():
            name, _, command = line.partition("\t")
            parsed.append((name, command))
        return parsed

    def names(self) -> list[str]:
        return [name for name, _ in self.calls()]

    def command(self, name: str) -> str:
        for called, command in self.calls():
            if called == name:
                return command
        raise AssertionError(f"ssh stub 没有收到 {name}：{self.names()}")

    def receipt(self) -> Path:
        return self.artifacts / "submit-receipts" / f"{RUN_ID}.json"

    def test_successful_empty_queue_goes_to_accounting(self) -> None:
        result = self.dispatch(squeue="\n", sacct="COMPLETED|0:0\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.names().count("squeue"), 1, "空队列被当成还在跑，白等预算")
        self.assertIn("sacct", self.names(), "空队列没有进入记账宽限")
        self.assertNotIn("scancel", self.names())
        self.assertTrue(self.receipt().is_file())

    def test_late_accounting_is_waited_for(self) -> None:
        result = self.dispatch(squeue="\n", sacct="\nCOMPLETED|0:0\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertGreaterEqual(self.names().count("sacct"), 2, "记账迟到被当成重投理由")

    def test_query_failure_is_unknown_not_done(self) -> None:
        result = self.dispatch(squeue="FAIL\nFAIL\nFAIL\n", sacct="COMPLETED|0:0\n")
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertNotIn("sacct", self.names(), "查询失败被当成了离队")
        self.assertNotIn("scancel", self.names())
        self.assertIn("unknown, not failed", result.stderr)
        self.assertTrue(self.receipt().is_file(), "未知也必须留下回执")

    def test_cap_and_archive_use_one_file_set(self) -> None:
        result = self.dispatch(squeue="\n", sacct="COMPLETED|0:0\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        measured, fetched = self.command("du"), self.command("fetch")
        for token in ("out", "run.env", "slurm-*.out", "slurm-*.err"):
            with self.subTest(token=token):
                self.assertIn(token, measured)
                self.assertIn(token, fetched)
        for token in ("--exclude=__pycache__", "--exclude='*.gguf'"):
            with self.subTest(exclude=token):
                self.assertIn(token, measured)
                self.assertIn(token, fetched)
        self.assertIn("KiB (cap", result.stdout, "限额没有按 KiB 比较")

    def test_over_cap_is_refused_before_fetching(self) -> None:
        result = self.dispatch(squeue="\n", sacct="COMPLETED|0:0\n", evidence_kib=2048, cap_mb=1)
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertNotIn("fetch", self.names(), "超限还是把证据拉回来了")
        self.assertIn("over the 1024 KiB cap", result.stderr)
