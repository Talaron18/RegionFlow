"""Resolve every name each generated probe uses. No injected imports."""
import ast, builtins, sys, textwrap
src = open(sys.argv[1]).read(); ast.parse(src)
ns = {}
exec(compile(src.replace('raise SystemExit(main())', 'pass'), 'p', 'exec'), ns)
pre = ns['PREAMBLE'].format(model='M', dtype='float32', tf32=False)  # verbatim

def defined(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store): out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: out.add(a.asname or a.name.split('.')[0])
        elif isinstance(n, ast.arg): out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            for m in ast.walk(n.optional_vars):
                if isinstance(m, ast.Name): out.add(m.id)
        elif isinstance(n, ast.comprehension):
            for m in ast.walk(n.target):
                if isinstance(m, ast.Name): out.add(m.id)
    return out

bad = False
for i, (title, _, body) in enumerate(ns['PROBES'], 1):
    tree = ast.parse(pre + textwrap.dedent(body))
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    missing = sorted(used - defined(tree) - set(dir(builtins)))
    print(f"probe {i}: {'OK' if not missing else 'UNDEFINED -> ' + str(missing)}")
    bad |= bool(missing)
sys.exit(1 if bad else 0)
