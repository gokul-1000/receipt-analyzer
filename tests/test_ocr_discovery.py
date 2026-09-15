"""
tests/test_ocr_discovery.py — Tests for Tesseract executable discovery logic.

All tests run without a real Tesseract binary by mocking filesystem and
subprocess calls.  No API key or OCR installation required.
"""

from __future__ import annotations

import os
import sys
import platform
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.ocr_service import (
    _WINDOWS_CANDIDATE_PATHS,
    _build_not_found_message,
    _is_executable,
    _verify_tesseract,
    find_tesseract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_isfile(valid_paths: set[str]):
    """Return a function that considers only *valid_paths* as existing files."""
    def _impl(path: str) -> bool:
        return path in valid_paths
    return _impl


# ---------------------------------------------------------------------------
# find_tesseract — TESSERACT_CMD environment variable
# ---------------------------------------------------------------------------

class TestFindTesseractEnvVar:
    def test_env_var_valid_path_is_used(self, tmp_path):
        """If TESSERACT_CMD points to an existing file, it should be returned."""
        fake_exe = tmp_path / "tesseract.exe"
        fake_exe.write_bytes(b"")  # create a real temporary file

        with patch.dict(os.environ, {"TESSERACT_CMD": str(fake_exe)}):
            result = find_tesseract()

        assert result == str(fake_exe)

    def test_env_var_nonexistent_path_falls_through(self, monkeypatch):
        """If TESSERACT_CMD points to a missing file, we fall through to PATH."""
        monkeypatch.setenv("TESSERACT_CMD", "/nonexistent/tesseract")

        # Also make PATH search fail so the result is None
        with patch("shutil.which", return_value=None), \
             patch("platform.system", return_value="Linux"):
            result = find_tesseract()

        assert result is None

    def test_env_var_empty_string_is_ignored(self, monkeypatch):
        """Empty TESSERACT_CMD should be treated as not set."""
        monkeypatch.setenv("TESSERACT_CMD", "   ")

        with patch("shutil.which", return_value=None), \
             patch("platform.system", return_value="Linux"):
            result = find_tesseract()

        assert result is None


# ---------------------------------------------------------------------------
# find_tesseract — PATH-based discovery
# ---------------------------------------------------------------------------

class TestFindTesseractPath:
    def test_found_on_path(self, monkeypatch):
        """shutil.which returns a path → find_tesseract returns it."""
        monkeypatch.delenv("TESSERACT_CMD", raising=False)

        with patch("shutil.which", return_value="/usr/bin/tesseract"):
            result = find_tesseract()

        assert result == "/usr/bin/tesseract"

    def test_not_on_path_linux_returns_none(self, monkeypatch):
        """On Linux with no PATH hit and no env var → None."""
        monkeypatch.delenv("TESSERACT_CMD", raising=False)

        with patch("shutil.which", return_value=None), \
             patch("platform.system", return_value="Linux"):
            result = find_tesseract()

        assert result is None


# ---------------------------------------------------------------------------
# find_tesseract — Windows default installation directory
# ---------------------------------------------------------------------------

class TestFindTesseractWindows:
    def test_windows_default_path_found(self, monkeypatch):
        """On Windows, when PATH fails, the first valid candidate is returned."""
        monkeypatch.delenv("TESSERACT_CMD", raising=False)

        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        with patch("shutil.which", return_value=None), \
             patch("platform.system", return_value="Windows"), \
             patch("os.path.isfile", _fake_isfile({default_path})):
            result = find_tesseract()

        assert result == default_path

    def test_windows_x86_fallback(self, monkeypatch):
        """x86 path is used when the 64-bit default is absent."""
        monkeypatch.delenv("TESSERACT_CMD", raising=False)

        x86_path = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

        with patch("shutil.which", return_value=None), \
             patch("platform.system", return_value="Windows"), \
             patch("os.path.isfile", _fake_isfile({x86_path})):
            result = find_tesseract()

        assert result == x86_path

    def test_windows_no_candidates_returns_none(self, monkeypatch):
        """When all Windows candidates are missing → None."""
        monkeypatch.delenv("TESSERACT_CMD", raising=False)

        with patch("shutil.which", return_value=None), \
             patch("platform.system", return_value="Windows"), \
             patch("os.path.isfile", return_value=False):
            result = find_tesseract()

        assert result is None

    def test_windows_candidates_list_is_non_empty(self):
        """Sanity check that the candidate list contains at least the standard paths."""
        assert any(
            "Program Files" in p and "Tesseract-OCR" in p
            for p in _WINDOWS_CANDIDATE_PATHS
        )


# ---------------------------------------------------------------------------
# find_tesseract — priority ordering
# ---------------------------------------------------------------------------

class TestFindTesseractPriority:
    def test_env_var_beats_path(self, tmp_path, monkeypatch):
        """TESSERACT_CMD is preferred over PATH even when PATH also works."""
        env_exe = tmp_path / "env_tesseract.exe"
        env_exe.write_bytes(b"")

        monkeypatch.setenv("TESSERACT_CMD", str(env_exe))

        with patch("shutil.which", return_value="/usr/bin/tesseract"):
            result = find_tesseract()

        assert result == str(env_exe)

    def test_path_beats_windows_defaults(self, monkeypatch):
        """PATH result is returned before checking Windows directories."""
        monkeypatch.delenv("TESSERACT_CMD", raising=False)
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        with patch("shutil.which", return_value="/usr/bin/tesseract"), \
             patch("platform.system", return_value="Windows"), \
             patch("os.path.isfile", _fake_isfile({default_path})):
            result = find_tesseract()

        assert result == "/usr/bin/tesseract"


# ---------------------------------------------------------------------------
# _verify_tesseract
# ---------------------------------------------------------------------------

class TestVerifyTesseract:
    def test_returns_true_on_success(self):
        """A zero exit-code response means Tesseract is working."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "tesseract 5.3.0\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            assert _verify_tesseract("/some/tesseract") is True

    def test_returns_false_on_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        with patch("subprocess.run", return_value=mock_result):
            assert _verify_tesseract("/some/tesseract") is False

    def test_returns_false_when_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _verify_tesseract("/missing/tesseract") is False

    def test_returns_false_on_timeout(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="tesseract", timeout=10)):
            assert _verify_tesseract("/some/tesseract") is False


# ---------------------------------------------------------------------------
# _build_not_found_message
# ---------------------------------------------------------------------------

class TestNotFoundMessage:
    def test_windows_message_contains_install_hint(self):
        with patch("platform.system", return_value="Windows"):
            msg = _build_not_found_message()
        assert "TESSERACT_CMD" in msg or "UB-Mannheim" in msg or "install" in msg.lower()

    def test_linux_message_contains_apt(self):
        with patch("platform.system", return_value="Linux"):
            msg = _build_not_found_message()
        assert "apt" in msg or "packages.txt" in msg

    def test_macos_message_contains_brew(self):
        with patch("platform.system", return_value="Darwin"):
            msg = _build_not_found_message()
        assert "brew" in msg


# ---------------------------------------------------------------------------
# _is_executable helper
# ---------------------------------------------------------------------------

class TestIsExecutable:
    def test_returns_true_for_real_file(self, tmp_path):
        f = tmp_path / "tess.exe"
        f.write_bytes(b"")
        assert _is_executable(str(f)) is True

    def test_returns_false_for_missing_file(self):
        assert _is_executable("/nonexistent/path/tesseract") is False

    def test_returns_false_for_empty_string(self):
        assert _is_executable("") is False

    def test_returns_false_for_directory(self, tmp_path):
        assert _is_executable(str(tmp_path)) is False
