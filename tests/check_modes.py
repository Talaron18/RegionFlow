"""Verify TransportAdapter accepts mode="meanflow" -- by parsing, not grep.

`grep -c meanflow transport/adapter.py` returned 4 on a file whose mode guard
still rejected it: the docstring and _gate had been updated while the
validation line had not, because a str.replace target missed by one character
(`{cfg.mode}` vs `{cfg.mode!r}`) and failed silently.

    python tests/check_modes.py transport/adapter.py
"""
import ast
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "transport/adapter.py"
tree = ast.parse(open(path).read())
cls = next(n for n in tree.body
           if isinstance(n, ast.ClassDef) and n.name == "TransportAdapter")
init = next(n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__")
guard = next(n for n in ast.walk(init)
             if isinstance(n, ast.Compare) and isinstance(n.ops[0], ast.NotIn))
allowed = set(ast.literal_eval(guard.comparators[0]))
print("modes accepted:", sorted(allowed))

gate = next(n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_gate")
gate_src = ast.get_source_segment(open(path).read(), gate)
ungated = '"pair", "meanflow"' in gate_src or "'pair', 'meanflow'" in gate_src

ok = allowed == {"interval", "pair", "meanflow"} and ungated
print("meanflow is ungated in _gate:", ungated)
print("PASS" if ok else "FAIL -- the file is stale or the patch missed")
sys.exit(0 if ok else 1)
