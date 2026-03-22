$ErrorActionPreference = "Stop"

$PythonSelector = if ($env:PYTHON_SELECTOR) { $env:PYTHON_SELECTOR } else { "py -3.11" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }

Invoke-Expression "$PythonSelector -m venv $VenvDir"
. "$VenvDir\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel
if (Test-Path .\requirements-lock.txt) {
  python -m pip install -r .\requirements-lock.txt
}
python -m pip install -e .[dev]
