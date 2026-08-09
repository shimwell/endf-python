"""The type stub, held to what the extension module actually exports.

A hand-written `.pyi` for a compiled module rots the moment someone adds a
method and forgets it — and it rots silently, because nothing imports it. This
parses the stub and compares the names it declares against the built module,
in both directions.

It checks names, not types: a type checker would have to run to verify those,
and the value here is catching the method that was added to `lib.rs` and never
written down. Skipped whole when the extension is not built.
"""

import ast
from pathlib import Path

import pytest

_endf = pytest.importorskip(
    "_endf", reason="the Rust extension module is not built; see test_rust_bindings.py"
)

STUB = Path(__file__).parent.parent / "crates" / "endf-py" / "_endf.pyi"


def parse_stub():
    """The stub's declared names: module-level, and per class."""
    tree = ast.parse(STUB.read_text())
    module_level = set()
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    members.add(item.target.id)
            classes[node.name] = members
            module_level.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_level.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_level.add(node.target.id)
    return module_level, classes


def public(names):
    return {n for n in names if not n.startswith("_")}


def test_the_stub_file_is_there():
    assert STUB.is_file(), f"{STUB} is missing"


def test_module_level_names_match():
    declared, _ = parse_stub()
    actual = public(dir(_endf))

    missing = actual - declared
    assert not missing, (
        f"the extension exports {sorted(missing)}, which the stub does not "
        f"declare. Add them to {STUB.name}."
    )

    extra = public(declared) - actual
    assert not extra, (
        f"the stub declares {sorted(extra)}, which the extension does not "
        f"export. Delete them from {STUB.name}."
    )


def test_class_members_match():
    _, classes = parse_stub()

    for name in sorted(public(dir(_endf))):
        obj = getattr(_endf, name)
        if not isinstance(obj, type):
            continue
        assert name in classes, f"{name} is a class but the stub has no ClassDef"

        actual = public(dir(obj))
        declared = public(classes[name])

        missing = actual - declared
        assert not missing, (
            f"{name} has {sorted(missing)}, which the stub does not declare"
        )

        extra = declared - actual
        assert not extra, (
            f"the stub declares {name}.{sorted(extra)}, which does not exist"
        )


def test_the_stub_is_installed_beside_the_module():
    # Finding `_endf.pyi` in the source tree is not the same as a consumer
    # getting it. maturin turns that file into a PEP 561 package — the
    # extension, `__init__.pyi` and a `py.typed` marker together — but only
    # because the file is there, which is exactly the kind of implicit
    # behaviour worth pinning. A wheel built without it works perfectly and
    # silently carries no types at all.
    installed = Path(_endf.__file__).parent

    assert (installed / "__init__.pyi").is_file(), (
        f"no stub installed at {installed}: the wheel was built without one, "
        "so nothing using this module gets types"
    )
    assert (installed / "py.typed").is_file(), (
        f"no py.typed marker at {installed}: type checkers ignore the stub without it"
    )


def test_the_installed_stub_is_the_one_in_the_tree():
    # Otherwise the tests above check a stub nobody ships.
    installed = Path(_endf.__file__).parent / "__init__.pyi"
    assert installed.read_text() == STUB.read_text(), (
        "the installed stub differs from crates/endf-py/_endf.pyi; the wheel "
        "is stale, so rebuild it before trusting the checks above"
    )
