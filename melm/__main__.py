"""MELM CLI entry point: python -m melm <command>"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from melm.appliance.provisioning import load_secrets_env

load_secrets_env()

from scripts.local_assistant_os_cli import main

if __name__ == "__main__":
    main()
