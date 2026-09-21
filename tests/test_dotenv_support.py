import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
KEY = "OPENROUTER_API_KEY"
SETUP_ERROR = (
    "Установите OPENROUTER_API_KEY в окружении Windows (Окружения) "
    "или в .env, см .env.example"
)


def isolated_environment():
    environment = os.environ.copy()
    environment.pop(KEY, None)
    environment.pop("PYTHON_DOTENV_DISABLED", None)
    environment["PYTHONPATH"] = str(ROOT)
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


@pytest.mark.parametrize("module", ["forecast_experiment", "live_trader"])
@pytest.mark.parametrize("existing", [None, "", "test-env-key"])
def test_dotenv_fallback_and_environment_priority(tmp_path, module, existing):
    # Windows editors may save UTF-8 with a BOM. Use only fake credentials.
    (tmp_path / ".env").write_text(f"{KEY}=test-file-key\n", encoding="utf-8-sig")
    environment = isolated_environment()
    if existing is not None:
        environment[KEY] = existing
    expected = existing or "test-file-key"
    process = subprocess.run(
        [sys.executable, "-c",
         "import importlib, os, sys; importlib.import_module(sys.argv[1]); "
         "assert os.getenv('OPENROUTER_API_KEY') == sys.argv[2]",
         module, expected],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        encoding="utf-8", timeout=20,
    )
    assert process.returncode == 0, process.stderr
    assert "test-file-key" not in process.stdout + process.stderr
    assert "test-env-key" not in process.stdout + process.stderr


@pytest.mark.parametrize("module", ["forecast_experiment", "live_trader"])
@pytest.mark.parametrize("empty_dotenv", [False, True])
def test_missing_key_exits_one_with_setup_instructions(tmp_path, module, empty_dotenv):
    if empty_dotenv:
        (tmp_path / ".env").write_text(f"{KEY}=\n", encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(ROOT / f"{module}.py"), "--asset", "BTC"],
        cwd=tmp_path, env=isolated_environment(), capture_output=True, text=True,
        encoding="utf-8", timeout=20,
    )
    assert process.returncode == 1
    assert SETUP_ERROR in process.stderr
    assert "Traceback" not in process.stderr
