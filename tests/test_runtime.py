from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from ja2zh_srt import main, models
from ja2zh_srt.sakura import ManagedSakuraServer, SakuraServerConfig
from ja2zh_srt.translate import TranslationError


class RuntimeCheckTests(unittest.TestCase):
    @patch("ja2zh_srt.models.subprocess.run")
    @patch("ja2zh_srt.models.shutil.which", return_value="/bin/llama-server")
    def test_cpu_only_llama_server_fails(self, _which: MagicMock, run: MagicMock) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = "Available devices:\n  CPU: Generic CPU\n"
        run.return_value.stderr = ""
        output = io.StringIO()
        with redirect_stdout(output):
            ready = models._print_llama_server()
        self.assertFalse(ready)
        self.assertIn("cannot see a CUDA device", output.getvalue())

    @patch("ja2zh_srt.models.ctranslate2.get_cuda_device_count", return_value=0)
    def test_missing_cuda_device_fails(self, _device_count: MagicMock) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            ready = models._print_cuda()
        self.assertFalse(ready)
        self.assertIn("CUDA device not visible", output.getvalue())

    @patch("ja2zh_srt.models.print_status", return_value=False)
    def test_check_returns_failure(self, _print_status: MagicMock) -> None:
        result = CliRunner().invoke(main, ["check"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("runtime is not ready", result.output)


class SakuraServerTests(unittest.TestCase):
    @patch("ja2zh_srt.sakura.subprocess.Popen")
    @patch("ja2zh_srt.sakura.ManagedSakuraServer._wait_until_ready")
    def test_startup_failure_stops_server(
        self, wait_until_ready: MagicMock, popen: MagicMock
    ) -> None:
        wait_until_ready.side_effect = TranslationError("startup failed")
        popen.return_value.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_path = Path(temporary_dir) / "model.gguf"
            model_path.touch()
            server = ManagedSakuraServer(
                SakuraServerConfig(
                    model_path=model_path,
                    port=12345,
                    log_path=Path(temporary_dir) / "server.log",
                )
            )
            with self.assertRaisesRegex(TranslationError, "startup failed"):
                server.__enter__()
            popen.return_value.terminate.assert_called_once()
            self.assertIsNone(server._process)
            self.assertIsNone(server._log_file)


if __name__ == "__main__":
    unittest.main()
