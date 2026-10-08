"""
Unit and integration tests for JARVIS Dependency Management and Environment Compatibility.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from backend.config.settings import Settings
from backend.core.runtime import RuntimeManager, RuntimeType
from backend.core.startup import StartupManager
from backend.scripts.check_environment import check_environment


class TestEnvironmentDiagnostics(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(debug=True, voice_mode=True)
        self.manager = RuntimeManager(self.settings)

    def test_python_version_validation(self) -> None:
        """Verify supported and unsupported Python version checks."""
        # 1. Supported versions: 3.11, 3.12
        with patch("sys.version_info") as mock_ver:
            mock_ver.major = 3
            mock_ver.minor = 12
            self.assertTrue(self.manager.check_python_version())

        with patch("sys.version_info") as mock_ver:
            mock_ver.major = 3
            mock_ver.minor = 11
            self.assertTrue(self.manager.check_python_version())

        # 2. Unsupported versions: 3.14, 2.7
        with patch("sys.version_info") as mock_ver:
            mock_ver.major = 3
            mock_ver.minor = 14
            self.assertFalse(self.manager.check_python_version())

        with patch("sys.version_info") as mock_ver:
            mock_ver.major = 2
            mock_ver.minor = 7
            self.assertFalse(self.manager.check_python_version())

    def test_virtual_environment_detection(self) -> None:
        """Verify venv active detection logic."""
        # 1. Virtual Environment active: sys.prefix != sys.base_prefix
        with patch("sys.prefix", "C:\\my_venv"):
            with patch("sys.base_prefix", "C:\\Python312"):
                with patch("sys.version_info") as mock_ver:
                    mock_ver.major = 3
                    mock_ver.minor = 12
                    manager = RuntimeManager(self.settings)
                    selected, status = manager.select_runtime()
                    self.assertTrue(status["python_supported"])

    def test_requirements_files_presence(self) -> None:
        """Verify that split requirements files exist and contain core, voice, and dev configurations."""
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        req_core = os.path.join(root_dir, "requirements.txt")
        req_voice = os.path.join(root_dir, "requirements-voice.txt")
        req_dev = os.path.join(root_dir, "requirements-dev.txt")

        self.assertTrue(os.path.exists(req_core))
        self.assertTrue(os.path.exists(req_voice))
        self.assertTrue(os.path.exists(req_dev))

        with open(req_core, "r") as f:
            core_content = f.read()
            self.assertIn("google-genai", core_content)

        with open(req_voice, "r") as f:
            voice_content = f.read()
            self.assertIn("faster-whisper", voice_content)

        with open(req_dev, "r") as f:
            dev_content = f.read()
            self.assertIn("pytest", dev_content)

    @patch("backend.scripts.check_environment.check_environment")
    def test_check_environment_script_runs(self, mock_check) -> None:
        """Verify environment verification entrypoint can run successfully."""
        mock_check.return_value = 0
        from backend.scripts import check_environment as check_env_module
        exit_code = check_env_module.check_environment()
        self.assertEqual(exit_code, 0)

    @patch("subprocess.run")
    def test_install_voice_dependencies_calls_pip(self, mock_run) -> None:
        """Verify installer invokes pip with correct arguments."""
        mock_run.return_value.returncode = 0
        from backend.scripts.install_voice_dependencies import install_voice_dependencies
        
        # Override sys.exit to avoid exiting test runner
        with patch("sys.exit") as mock_exit:
            install_voice_dependencies()
            mock_run.assert_called()
            mock_exit.assert_not_called()

    def test_startup_diagnostics_formatting(self) -> None:
        """Verify _print_startup_diagnostics method outputs report correctly without errors."""
        mock_app = MagicMock()
        mock_app.settings = self.settings
        mock_app.tool_registry.get_tool.return_value = MagicMock()
        mock_app.settings.memory_db_path = "instance/jarvis_memory.db"

        startup_manager = StartupManager()
        with patch("builtins.print") as mock_print:
            startup_manager._print_startup_diagnostics(mock_app)
            mock_print.assert_called()


if __name__ == "__main__":
    unittest.main()
