"""Cut a release: one number, every place it has to appear, one tag.

    python -m scripts.release 0.1.2

The version lives in three files — the package, the project metadata and the
NAS compose file that pins it — and a test refuses to pass if they disagree.
This writes all three, commits, and tags; pushing the tag is what starts the
build on GitHub, and that is left to you so nothing leaves the machine
without a look at `git show`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EDITS = (
    (ROOT / "app" / "__init__.py", r'^__version__ = "[^"]+"', '__version__ = "{v}"'),
    (ROOT / "pyproject.toml", r'^version = "[^"]+"', 'version = "{v}"'),
    (
        ROOT / "deploy" / "docker-compose.nas.yml",
        r"(image: ghcr\.io/themechanic-dev/trance-autodj:)[0-9][^\s]*",
        r"\g<1>{v}",
    ),
)


def _git(*args: str) -> None:
    # The arguments are ours: file names from EDITS and a version that has
    # already matched \d+\.\d+\.\d+. Nothing here comes from outside.
    subprocess.run(["git", *args], cwd=ROOT, check=True)  # noqa: S603, S607


def main(argv: list[str]) -> int:
    version = argv[1] if len(argv) == 2 else ""  # noqa: PLR2004 - "one argument"
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        print(__doc__)
        return 2

    for path, pattern, replacement in EDITS:
        text = path.read_text(encoding="utf-8")
        new, count = re.subn(pattern, replacement.format(v=version), text, flags=re.M)
        if count != 1:
            print(f"{path.relative_to(ROOT)}: expected one match, found {count}")
            return 1
        path.write_text(new, encoding="utf-8")
        print(f"  {path.relative_to(ROOT)} -> {version}")

    files = [str(p.relative_to(ROOT)) for p, _, _ in EDITS]
    _git("add", *files)
    _git("commit", "-q", "-m", version)
    _git("tag", "-a", f"v{version}", "-m", f"Trance AutoDJ {version}")
    print(
        f"\ncommitted and tagged v{version}. To publish:\n\n    git push origin main v{version}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
