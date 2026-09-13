"""Patch TransportAdapter's mode guard in place, and verify at RUNTIME.

Run this in the container that will do the training:

    python tests/fix_adapter_mode.py

Checking a file on disk is not the same as checking what Python imports --
a stale copy on sys.path, a shadowed package, or a re-extracted archive all
produce a file that greps correctly and a module that does not. This imports
TransportAdapter, prints the file it actually came from, and instantiates
mode="meanflow" for real.
"""
import ast
import re
import sys
from pathlib import Path

import transport.adapter as A

path = Path(A.__file__)
print(f"python imports TransportAdapter from:\n  {path}")
src = path.read_text()

tree = ast.parse(src)
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "TransportAdapter")
init = next(n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__")
guard = next(n for n in ast.walk(init)
             if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.NotIn))
allowed = set(ast.literal_eval(guard.comparators[0]))
print(f"guard currently accepts: {sorted(allowed)}")

if allowed != {"interval", "pair", "meanflow"}:
    print("patching...")
    new_src = re.sub(
        r'if cfg\.mode not in \([^)]*\):\s*\n\s*raise ValueError\((?:[^()]|\([^()]*\))*\)',
        'if cfg.mode not in ("interval", "pair", "meanflow"):\n'
        '            raise ValueError(\n'
        '                "mode must be \'interval\', \'pair\' or \'meanflow\', got "\n'
        '                f"{cfg.mode!r}")',
        src, count=1)
    if new_src == src:
        print("  regex did not match; edit the guard by hand:")
        print("    transport/adapter.py, TransportAdapter.__init__")
        print('    if cfg.mode not in ("interval", "pair", "meanflow"):')
        sys.exit(1)
    path.write_text(new_src)
    print("  patched. RESTART python -- the old module is already imported.")
    sys.exit(2)

# _gate must treat meanflow as ungated too.
gate = next(n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_gate")
gate_src = ast.get_source_segment(src, gate)
ungated = '"pair", "meanflow"' in gate_src or "'pair', 'meanflow'" in gate_src
print(f"_gate treats meanflow as ungated: {ungated}")
if not ungated:
    print("  FAIL: _gate still gates meanflow on (t - r); the anchors will\n"
          "  produce zero gradient. Fix _gate before training.")
    sys.exit(1)

# Runtime proof, not a file inspection.
try:
    A.AdapterConfig(mode="meanflow")
    print("\nPASS: AdapterConfig(mode='meanflow') constructs, and the module\n"
          "python imports is the patched one.")
except Exception as exc:  # noqa: BLE001
    print(f"\nFAIL at runtime: {type(exc).__name__}: {exc}")
    sys.exit(1)
