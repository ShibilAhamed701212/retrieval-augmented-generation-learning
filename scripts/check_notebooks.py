"""Static checks for the notebooks, runnable without API keys or network access.

Checks that every code cell is valid Python (IPython magics and shell lines are
skipped) and, with --imports, that every import in the notebooks resolves
against the installed requirements.
"""

import argparse
import ast
import importlib
import json
import sys
import warnings
from pathlib import Path

# Installed by its own notebook cell (it pulls in torch), not by requirements.txt
OPTIONAL_PACKAGES = {"ragatouille"}

ROOT = Path(__file__).resolve().parent.parent


def code_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines = [
            "" if line.lstrip().startswith(("!", "%")) else line
            for line in source.splitlines()
        ]
        yield index, "\n".join(lines)


def check_import(module, name):
    imported = importlib.import_module(module)
    if name is not None and not hasattr(imported, name):
        # `from package import submodule`
        importlib.import_module(f"{module}.{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports", action="store_true", help="also check that imports resolve"
    )
    args = parser.parse_args()

    errors = []
    imports = {}
    notebooks = sorted(ROOT.glob("*.ipynb"))
    for path in notebooks:
        for index, source in code_cells(path):
            location = f"{path.name} cell {index}"
            try:
                tree = ast.parse(source, filename=location)
            except SyntaxError as exc:
                errors.append(f"{location}: {exc.msg} (line {exc.lineno})")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.setdefault((node.module, alias.name), location)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.setdefault((alias.name, None), location)

    if args.imports:
        warnings.simplefilter("ignore")
        for (module, name), location in sorted(imports.items(), key=str):
            if module.split(".")[0] in OPTIONAL_PACKAGES:
                continue
            try:
                check_import(module, name)
            except Exception as exc:  # report every failure, not just the first
                target = f"{module}.{name}" if name else module
                errors.append(f"{location}: cannot import {target}: {exc}")

    for error in errors:
        print(error, file=sys.stderr)
    print(
        f"Checked {len(notebooks)} notebooks, {len(imports)} imports: "
        f"{len(errors)} problem(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
