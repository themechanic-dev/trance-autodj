#!/usr/bin/env python3
"""Run the environment check and print a report.

    python scripts/preflight.py
    python scripts/preflight.py --json
    python scripts/preflight.py --config /etc/trance-autodj/config.yaml

Exit code 0 when nothing blocking was found, 1 otherwise, so it can be used
as a container healthcheck or in a systemd ExecStartPre.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import load_config
from app.core.paths import Paths
from app.services.system.preflight import format_report, run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that this machine can run Trance AutoDJ")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"\033[31m✖ configuration is invalid\033[0m\n{exc}", file=sys.stderr)
        return 1

    paths = Paths.from_config(cfg)
    report = run_preflight(cfg, paths)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    else:
        colour = not args.no_color and sys.stdout.isatty()
        print(format_report(report, color=colour))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
