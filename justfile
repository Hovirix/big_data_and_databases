default:
    @just --list

sync:
    uv sync --locked

view:
    uv run --locked python -m src.view_data

audit:
    uv run --locked python -m src.evaluate --check-only

test:
    uv run --locked python -m unittest discover

overview:
    uv run --locked python -m src.overview

prepare:
    uv run --locked python -m src.prepare

train:
    uv run --locked python -m src.train

evaluate:
    uv run --locked python -m src.evaluate

# Code + tests + reproduction settings; excludes data, outputs and caches.
package destination="telco-code.zip":
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --locked python - "{{destination}}" <<'PY'
    from pathlib import Path
    import sys
    from zipfile import ZIP_DEFLATED, ZipFile

    files = [
        *Path("src").glob("*.py"), *Path("tests").glob("*.py"),
        "README.md", "AGENTS.md", "pyproject.toml", "uv.lock",
        "flake.nix", "flake.lock", ".envrc", ".gitignore", "justfile",
    ]
    with ZipFile(sys.argv[1], "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path)
    PY
