"""Scope-aware undefined-name check.

    python tests/check_names.py scripts/*.py transport/*.py

WHY SCOPE-AWARE. The first version collected every binding anywhere in a file
into one flat set, so a function-local `import torch` made `torch` look
available to every other function. That is exactly how `encode_null` shipped
using torch when only `cmd_train` imported it -- the checker passed, and the
run died after loading a 2B-parameter model.

Model: module scope = top-level bindings. Function scope = module scope +
its own bindings + its parameters + enclosing function scopes. Comprehension
and with/except targets count as bindings. Class bodies are treated as
functions, which is slightly wrong for methods but errs toward reporting.

Conservative by construction: star-imports, globals() tricks and runtime
injection will produce false positives. Read them rather than suppressing.
"""
import ast
import builtins
import sys

BUILTIN = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__"}


def bindings(node, recurse_into_functions=False):
    """Names bound directly in this node's body (not inside nested functions
    unless asked)."""
    out = set()

    def walk(n, top=False):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                out.add(child.name)
                if recurse_into_functions:
                    walk(child)
                continue
            if isinstance(child, ast.Lambda):
                continue          # its own scope; handled in loads()
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                out.add(child.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for a in child.names:
                    out.add(a.asname or a.name.split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                out.add(child.name)
            elif isinstance(child, ast.Global) or isinstance(child, ast.Nonlocal):
                out.update(child.names)
            walk(child)

    walk(node, top=True)
    return out


def _lambda_args(node):
    a = node.args
    return {arg.arg for arg in (a.posonlyargs + a.args + a.kwonlyargs
                                + ([a.vararg] if a.vararg else [])
                                + ([a.kwarg] if a.kwarg else []))}


def _comp_targets(node):
    out = set()
    for gen in node.generators:
        for m in ast.walk(gen.target):
            if isinstance(m, ast.Name):
                out.add(m.id)
    return out


def loads(node):
    """Names loaded in this node, excluding nested function bodies.

    Lambdas and comprehensions carry their own bindings, so their parameters
    and targets are subtracted rather than reported. Without this the checker
    flags every `lambda z_, t_, r_: ...` in the project, which is noise that
    teaches people to ignore it.
    """
    out = set()

    def walk(n, bound):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                continue
            if isinstance(child, ast.Lambda):
                walk(child, bound | _lambda_args(child))
                continue
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp,
                                  ast.GeneratorExp)):
                walk(child, bound | _comp_targets(child))
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in bound:
                    out.add(child.id)
            walk(child, bound)

    walk(node, set())
    return out


def check_file(path):
    tree = ast.parse(open(path).read())
    module_scope = bindings(tree) | BUILTIN
    problems = []

    def visit(node, enclosing, label):
        local = bindings(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            for arg in (a.posonlyargs + a.args + a.kwonlyargs
                        + ([a.vararg] if a.vararg else [])
                        + ([a.kwarg] if a.kwarg else [])):
                local.add(arg.arg)
        scope = enclosing | local
        for name in sorted(loads(node) - scope):
            problems.append(f"{label}: {name}")
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                visit(child, scope, f"{label}.{child.name}"
                      if label != "<module>" else child.name)

    visit(tree, module_scope, "<module>")
    return problems


bad = 0
for path in sys.argv[1:]:
    try:
        problems = check_file(path)
    except SyntaxError as exc:
        print(f"{path}: SYNTAX ERROR {exc}")
        bad += 1
        continue
    if problems:
        print(f"{path}:")
        for p in problems:
            print(f"  UNDEFINED IN SCOPE  {p}")
        bad += 1
print(f"\n{len(sys.argv) - 1} file(s) checked, {bad} with problems")
sys.exit(1 if bad else 0)
