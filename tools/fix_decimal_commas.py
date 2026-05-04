"""Convert pt-BR comma decimals (1,234) to JSON-valid dot decimals (1.234).

Useful when the Unity saver runs on a Brazilian/European locale and
serializes numbers with `,` instead of `.`. Run once over the captures
folder before tools/scan_pipeline.py.

The regex `(?<=\\d),(?=\\d)` only matches a comma that is *tightly* between
two digits (no whitespace), so JSON array separators like `[0, 1, 2]` and
property separators like `},\\n` are left untouched.

Examples
--------
Before::

    "fx": 869,222600,
    "pose_t_wc": [0,227021, 1,676882, -1,653543]

After::

    "fx": 869.222600,
    "pose_t_wc": [0.227021, 1.676882, -1.653543]

Usage
-----
::

    # Preview which files would change without writing anything
    python tools/fix_decimal_commas.py teste/meu_quarto --dry-run

    # Apply in place
    python tools/fix_decimal_commas.py teste/meu_quarto
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")


def fix_text(txt: str) -> str:
    return DECIMAL_COMMA.sub(".", txt)


def fix_file(path: Path) -> bool:
    txt = path.read_text(encoding="utf-8")
    fixed = fix_text(txt)
    if fixed == txt:
        return False
    path.write_text(fixed, encoding="utf-8")
    return True


def validate_json(path: Path) -> tuple[bool, str]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    ap = argparse.ArgumentParser(
        description="Convert pt-BR comma decimals to dot decimals in JSON files."
    )
    ap.add_argument("folder", help="Pasta com *.json a corrigir")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra quais arquivos seriam alterados, sem escrever.",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Tenta parsear cada arquivo como JSON depois de corrigir e reporta erros.",
    )
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Pasta nao encontrada: {folder}")

    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"Nenhum *.json em {folder}")
        return

    changed = 0
    errors = 0
    for p in files:
        if args.dry_run:
            txt = p.read_text(encoding="utf-8")
            if DECIMAL_COMMA.search(txt):
                print(f"[dry-run] would fix: {p.name}")
                changed += 1
        else:
            if fix_file(p):
                print(f"fixed: {p.name}")
                changed += 1
            if args.validate:
                ok, err = validate_json(p)
                if not ok:
                    print(f"  WARN: still invalid JSON: {err}")
                    errors += 1

    suffix = "que precisam corrigir" if args.dry_run else "atualizados"
    print(f"\n{changed} arquivos {suffix} (de {len(files)}).")
    if args.validate and errors:
        print(f"{errors} arquivo(s) ainda nao parseiam como JSON.")


if __name__ == "__main__":
    main()
