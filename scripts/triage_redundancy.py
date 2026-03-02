#!/usr/bin/env python3
"""Test redundancy triage with empirical deselection validation.

Outputs:
- inventory.csv
- coverage_matrix.csv
- coverage_summary.json
- mutation_matrix.csv
- mutation_summary.json
- branch_equiv_report.csv
- branch_equiv_summary.json
- branch_equiv_report.md
- confidence_gate_matrix.csv
- candidate_validation.csv
- candidate_validation.md
- candidate_validation_summary.json
- strict_gate.csv (when --strict-delete-gate is enabled)
- strict_gate_summary.json (when --strict-delete-gate is enabled)
"""

from __future__ import annotations

import argparse
import ast
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestMeta:
    nodeid: str
    file: str
    class_name: str
    test_name: str
    entrypoint: str
    intent: str
    assertion_types: set[str]
    assert_count: int = 0
    is_parametrized: bool = False
    src_tokens: frozenset = frozenset()  # normalised token set for Jaccard similarity


@dataclass(frozen=True)
class MutationProbe:
    probe_id: str
    file: str
    old: str
    new: str


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: int = 900) -> dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "runtime_ms": (time.time() - t0) * 1000.0,
            "output": (proc.stdout or "") + (proc.stderr or ""),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        out = ""
        if exc.stdout:
            out += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="ignore")
        if exc.stderr:
            out += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="ignore")
        out += f"\n[TIMEOUT] command exceeded {timeout}s: {' '.join(cmd)}"
        return {
            "returncode": 124,
            "runtime_ms": (time.time() - t0) * 1000.0,
            "output": out,
            "timed_out": True,
        }
    except FileNotFoundError as exc:
        return {
            "returncode": 127,
            "runtime_ms": (time.time() - t0) * 1000.0,
            "output": f"[ENOENT] {exc}",
            "timed_out": False,
        }


def resolve_python_exe(root: Path, raw: str) -> str:
    """Resolve interpreter path while preserving common shell-style usage.

    - Absolute paths are used as-is.
    - Relative paths containing a separator are resolved from repo root.
    - Bare commands (e.g. `python3`) are resolved via PATH.
    """
    p = Path(raw)
    if p.is_absolute():
        return str(p)

    if "/" in raw or "\\" in raw:
        candidate = (root / p).resolve()
        if candidate.exists():
            return str(candidate)

    which = shutil.which(raw)
    return which or raw


def resolve_optional_path(root: Path, raw: str) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def extract_calls(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            name = dotted_name(n.func)
            if name:
                out.add(name)
    return out


def infer_assertion_types(fn: ast.AST, calls: set[str], src: str) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.With):
            for item in n.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and dotted_name(ctx.func).endswith("raises"):
                    out.add("exception")

    if "isinstance" in calls:
        out.add("type_check")
    if "np.testing.assert_array_equal" in calls:
        out.add("array_equality")
    if "simplify" in calls:
        out.add("topology_equality")
    if ".dtype" in src:
        out.add("dtype_contract")
    if ".flags.writeable" in src:
        out.add("mutability_contract")
    if "len(" in src:
        out.add("length_contract")
    if "assert" in src and not out:
        out.add("general_assert")
    return out


def count_assertions(fn: ast.AST) -> int:
    """Count explicit assert statements in a test function."""
    return sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assert))


def detect_parametrized(fn: ast.AST) -> bool:
    """Return True if the function has a @pytest.mark.parametrize decorator."""
    decorators = getattr(fn, "decorator_list", [])
    for decorator in decorators:
        if isinstance(decorator, ast.Call):
            name = dotted_name(decorator.func)
        else:
            name = dotted_name(decorator)
        if "parametrize" in name:
            return True
    return False


def tokenize_normalized(src: str) -> frozenset:
    """Return a frozenset of normalised tokens suitable for Jaccard similarity.

    Literals are collapsed to sentinel strings so that tests differing only in
    concrete values (e.g. array shapes) still register as similar structure.
    """
    src = re.sub(r'"[^"]*"', "STR", src)
    src = re.sub(r"'[^']*'", "STR", src)
    src = re.sub(r"\b\d+\b", "NUM", src)
    tokens = re.findall(r"[a-zA-Z_]\w*", src)
    stopwords = {
        "self", "def", "return", "import", "from", "assert", "for", "in",
        "if", "not", "and", "or", "True", "False", "None", "with", "as",
        "class", "pass", "raise", "else", "elif", "try", "except",
    }
    return frozenset(t for t in tokens if len(t) > 2 and t not in stopwords)


def jaccard_sim(a: frozenset, b: frozenset) -> float:
    union_size = len(a | b)
    return len(a & b) / union_size if union_size else 1.0


def infer_entrypoint(calls: set[str], src: str, file_fallback: str = "", class_fallback: str = "") -> str:
    # Generic fallback: no repo-specific API name detection.
    # When a file_fallback is provided (the test file's repo-relative path),
    # tests are clustered by file/class to keep grouping broad enough to
    # surface overlap candidates in API-style suites.
    # To add project-specific entrypoint detection, extend this function with
    # your own if/elif checks on `calls` (the set of function names called in
    # the test body) or on `src` (the raw test source text).
    if file_fallback:
        cls = class_fallback or "<module>"
        return f"{file_fallback}::{cls}"
    if class_fallback:
        return class_fallback
    return "unknown"


def infer_intent(test_name: str, entrypoint: str, assertions: set[str], src: str) -> str:
    low = test_name.lower()
    if "version" in low:
        return "introspection"
    if "exception" in assertions:
        return "error_semantics"
    if "mutability_contract" in assertions or "dtype_contract" in assertions:
        return "shape_dtype_contract"
    if "array_equality" in assertions or "topology_equality" in assertions:
        return "parity_equivalence"
    # Mock/isolation tests
    if "monkeypatch" in low or "mock" in low or "patch" in low:
        return "mock_isolation"
    # Lifecycle/cleanup tests
    if "__del__" in low or "cleanup" in low or "teardown" in low or "lifecycle" in low:
        return "lifecycle_contract"
    # FFI/C-library tests
    if "cdll" in low or "ctypes" in low or "ffi" in low or "library" in low or "lib_stream" in low:
        return "ffi_contract"
    # Check source for monkeypatch usage
    if "monkeypatch" in src.lower():
        return "mock_isolation"
    if "ctypes" in src or "CDLL" in src or "_lib_stream" in src:
        return "ffi_contract"
    if "__del__" in src:
        return "lifecycle_contract"
    return "shape_dtype_contract"


def parse_test_metadata(root: Path, suite_files: list[str]) -> list[TestMeta]:
    tests: list[TestMeta] = []
    for rel in suite_files:
        path = (root / rel).resolve()
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        mod = ast.parse(src, filename=str(path))
        stack: list[str] = []

        class V(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                if not node.name.startswith("test_"):
                    return
                cls = stack[-1] if stack else ""
                try:
                    relpath = str(path.relative_to(root))
                except ValueError:
                    relpath = str(path)
                nodeid = f"{relpath}::{cls}::{node.name}" if cls else f"{relpath}::{node.name}"
                calls = extract_calls(node)
                fn_src = ast.get_source_segment(src, node) or ""
                assertions = infer_assertion_types(node, calls, fn_src)
                entry = infer_entrypoint(calls, fn_src, file_fallback=relpath, class_fallback=cls)
                intent = infer_intent(node.name, entry, assertions, fn_src)
                tests.append(
                    TestMeta(
                        nodeid=nodeid,
                        file=relpath,
                        class_name=cls,
                        test_name=node.name,
                        entrypoint=entry,
                        intent=intent,
                        assertion_types=assertions,
                        assert_count=count_assertions(node),
                        is_parametrized=detect_parametrized(node),
                        src_tokens=tokenize_normalized(fn_src),
                    )
                )

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                if not node.name.startswith("test_"):
                    return
                cls = stack[-1] if stack else ""
                try:
                    relpath = str(path.relative_to(root))
                except ValueError:
                    relpath = str(path)
                nodeid = f"{relpath}::{cls}::{node.name}" if cls else f"{relpath}::{node.name}"
                calls = extract_calls(node)
                fn_src = ast.get_source_segment(src, node) or ""
                assertions = infer_assertion_types(node, calls, fn_src)
                entry = infer_entrypoint(calls, fn_src, file_fallback=relpath, class_fallback=cls)
                intent = infer_intent(node.name, entry, assertions, fn_src)
                tests.append(
                    TestMeta(
                        nodeid=nodeid,
                        file=relpath,
                        class_name=cls,
                        test_name=node.name,
                        entrypoint=entry,
                        intent=intent,
                        assertion_types=assertions,
                        assert_count=count_assertions(node),
                        is_parametrized=detect_parametrized(node),
                        src_tokens=tokenize_normalized(fn_src),
                    )
                )

        V().visit(mod)

    tests.sort(key=lambda t: t.nodeid)
    return tests


def ensure_numba_stub(out_dir: Path) -> Path:
    stub_root = out_dir / "_runtime_stubs"
    numba_dir = stub_root / "numba"
    numba_dir.mkdir(parents=True, exist_ok=True)
    code = '''"""Audit runtime stub for numba"""

def njit(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    def _decorator(fn):
        return fn
    return _decorator

jit = njit
'''
    (numba_dir / "__init__.py").write_text(code, encoding="utf-8")
    return stub_root


def discover_import_roots(base: Path) -> list[str]:
    """Return conventional import roots for a repository root.

    Supports both flat layout (`package/`) and src layout (`src/package/`).
    """
    roots: list[str] = []
    src_dir = base / "src"
    if src_dir.is_dir():
        roots.append(str(src_dir))
    roots.append(str(base))
    return unique_preserve(roots)


def build_runtime_env(
    root: Path,
    out_dir: Path,
    python_exe: str,
    *,
    allow_numba_stub: bool,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    # Merge user-provided environment variables first so they can influence
    # subsequent steps (e.g. NUMBA_DISABLE_JIT, PATH overrides).
    if extra_env:
        env.update(extra_env)
    parts = []
    if allow_numba_stub:
        probe = run_cmd([python_exe, "-c", "import numba"], cwd=root, env=env, timeout=60)
        if probe["returncode"] != 0:
            parts.append(str(ensure_numba_stub(out_dir)))
    parts.extend(discover_import_roots(root))
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(unique_preserve(parts))
    return env


def has_xdist_plugin(root: Path, python_exe: str, env: dict[str, str]) -> bool:
    probe = run_cmd(
        [
            python_exe,
            "-c",
            "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('xdist') else 1)",
        ],
        cwd=root,
        env=env,
        timeout=60,
    )
    return probe["returncode"] == 0


def run_suite(
    root: Path,
    python_exe: str,
    suite_files: list[str],
    env: dict[str, str],
    deselect: str | None,
    timeout: int,
    *,
    use_xdist: bool,
) -> dict[str, Any]:
    deselects = [deselect] if deselect else []
    return run_suite_multi(
        root,
        python_exe,
        suite_files,
        env,
        deselects,
        timeout,
        use_xdist=use_xdist,
    )


def run_suite_multi(
    root: Path,
    python_exe: str,
    suite_files: list[str],
    env: dict[str, str],
    deselects: list[str] | None,
    timeout: int,
    *,
    use_xdist: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="triage_pycache_") as td:
        cache_dir = Path(td) / ".pytest_cache"
        cmd = [
            python_exe,
            "-P",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-o",
            "pythonpath=",
            "-o",
            f"cache_dir={cache_dir}",
            "-q",
        ]
        if use_xdist:
            cmd.extend(["-n", "0"])
        cmd += suite_files
        for nodeid in deselects or []:
            cmd.append(f"--deselect={nodeid}")
        return run_cmd(cmd, cwd=root, env=env, timeout=timeout)


def resolve_and_validate_suite_paths(root: Path, raw_paths: list[str], *, arg_name: str) -> list[str]:
    """Resolve suite paths and enforce they are inside --root."""
    missing: list[str] = []
    outside: list[str] = []
    normalized: list[str] = []

    for raw in raw_paths:
        p = Path(raw)
        abs_path = p.resolve() if p.is_absolute() else (root / p).resolve()
        if not abs_path.exists():
            missing.append(raw)
            continue
        try:
            rel = abs_path.relative_to(root)
            normalized.append(str(rel))
        except ValueError:
            outside.append(str(abs_path))

    if missing or outside:
        lines = [f"Invalid {arg_name} path(s):"]
        if missing:
            lines.append(f"- missing: {', '.join(missing)}")
        if outside:
            lines.append(f"- outside --root ({root}): {', '.join(outside)}")
        lines.append("All suite paths must exist and be located under --root.")
        raise SystemExit("\n".join(lines))

    return normalized


def parse_ranked_by_nodeid(path: Path) -> dict[str, dict[str, str]]:
    if not path or not path.exists():
        return {}
    rows = read_csv_rows(path)
    return {r.get("test_nodeid", ""): r for r in rows if r.get("test_nodeid")}


def parse_inventory_assertions(path: Path) -> dict[str, set[str]]:
    if not path or not path.exists():
        return {}
    rows = read_csv_rows(path)
    out = {}
    for r in rows:
        nodeid = r.get("test_nodeid", "")
        if not nodeid:
            continue
        raw = r.get("assertion_types", "")
        out[nodeid] = {x for x in raw.split(";") if x}
    return out


def infer_test_status(returncode: int, output: str) -> str:
    if returncode == 0 and re.search(r"\b\d+\s+skipped\b", output):
        return "skipped"
    if returncode == 0:
        return "passed"
    return "failed"


def normalize_source_path_for_coverage(raw_path: str, root: Path) -> str:
    norm = raw_path.replace("\\", "/")
    root_norm = str(root).replace("\\", "/")
    if norm.startswith(root_norm + "/"):
        norm = norm[len(root_norm) + 1 :]
    if norm.startswith("./"):
        norm = norm[2:]
    return norm


def parse_coverage_json(json_path: Path, root: Path, source_prefix: str = "") -> tuple[set[str], set[str]]:
    if not json_path.exists():
        return set(), set()

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set(), set()

    files: dict[str, Any] = data.get("files", {})
    line_tokens: set[str] = set()
    branch_tokens: set[str] = set()

    for file_path, file_data in files.items():
        rel = normalize_source_path_for_coverage(file_path, root)
        if source_prefix and not rel.startswith(source_prefix):
            continue

        for line_no in file_data.get("executed_lines", []):
            line_tokens.add(f"L|{rel}|{line_no}")

        for branch in file_data.get("executed_branches", []):
            if isinstance(branch, list | tuple) and len(branch) == 2:
                branch_tokens.add(f"B|{rel}|{branch[0]}|{branch[1]}")

    return line_tokens, branch_tokens


def prepend_pythonpath(env: dict[str, str], *prefixes: str) -> dict[str, str]:
    out = dict(env)
    existing = out.get("PYTHONPATH", "")
    parts = [p for p in prefixes if p]
    if existing:
        parts.append(existing)
    out["PYTHONPATH"] = os.pathsep.join(parts)
    return out


def ensure_coverage_tool(
    root: Path,
    out_dir: Path,
    python_exe: str,
    env: dict[str, str],
    timeout: int,
) -> tuple[str, dict[str, str], str, str]:
    """Resolve interpreter that can execute coverage.py.

    Returns (python_executable, coverage_env, mode, status_note), where mode is one of:
    - system: requested interpreter already has coverage
    - target_bootstrap: coverage installed into runtime PYTHONPATH target dir
    - unavailable: coverage could not be made available
    """
    probe = run_cmd([python_exe, "-m", "coverage", "--version"], cwd=root, env=env, timeout=60)
    if probe["returncode"] == 0:
        return python_exe, env, "system", "coverage tool available from requested interpreter"

    tools_dir = out_dir / "_runtime_tools"
    target_dir = tools_dir / "coverage_site"
    tools_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    coverage_env = prepend_pythonpath(env, str(target_dir))
    probe_target = run_cmd([python_exe, "-m", "coverage", "--version"], cwd=root, env=coverage_env, timeout=60)
    if probe_target["returncode"] != 0:
        install = run_cmd(
            [
                python_exe,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--target",
                str(target_dir),
                "coverage",
            ],
            cwd=root,
            env=env,
            timeout=max(300, timeout),
        )
        if install["returncode"] != 0:
            return "", env, "unavailable", f"failed to install coverage: {install['output'][:300]}"

    verify = run_cmd([python_exe, "-m", "coverage", "--version"], cwd=root, env=coverage_env, timeout=60)
    if verify["returncode"] == 0:
        return python_exe, coverage_env, "target_bootstrap", "coverage installed in runtime PYTHONPATH target"

    return "", env, "unavailable", f"coverage verification failed: {verify['output'][:300]}"


def run_single_test_coverage(
    root: Path,
    nodeid: str,
    *,
    coverage_python: str,
    env: dict[str, str],
    timeout: int,
    tmp_dir: Path,
    source_prefix: str = "",
) -> dict[str, Any]:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid)
    cov_data = tmp_dir / f"{safe}.coverage"
    cov_json = tmp_dir / f"{safe}.json"

    run = run_cmd(
        [
            coverage_python,
            "-P",
            "-m",
            "coverage",
            "run",
            "--branch",
            f"--data-file={cov_data}",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-o",
            "pythonpath=",
            "-q",
            nodeid,
        ],
        cwd=root,
        env=env,
        timeout=timeout,
    )

    if cov_data.exists():
        _ = run_cmd(
            [
                coverage_python,
                "-m",
                "coverage",
                "json",
                f"--data-file={cov_data}",
                "-o",
                str(cov_json),
            ],
            cwd=root,
            env=env,
            timeout=timeout,
        )

    line_tokens, branch_tokens = parse_coverage_json(cov_json, root, source_prefix)
    status = infer_test_status(run["returncode"], run["output"])

    return {
        "test_nodeid": nodeid,
        "status": status,
        "runtime_ms": round(run["runtime_ms"], 3),
        "executed_line_count": len(line_tokens),
        "executed_branch_count": len(branch_tokens),
        "_line_tokens": line_tokens,
        "_branch_tokens": branch_tokens,
        "error": "" if run["returncode"] == 0 else run["output"][:1200],
    }


def collect_suite_coverage_union(
    root: Path,
    suite_files: list[str],
    *,
    coverage_python: str,
    env: dict[str, str],
    timeout: int,
    tmp_dir: Path,
    source_prefix: str = "",
) -> dict[str, Any]:
    if not suite_files:
        return {
            "status": "not_configured",
            "runtime_ms": 0.0,
            "line_tokens": set(),
            "branch_tokens": set(),
            "error": "",
        }

    cov_data = tmp_dir / "comparator.coverage"
    cov_json = tmp_dir / "comparator.json"
    run = run_cmd(
        [
            coverage_python,
            "-P",
            "-m",
            "coverage",
            "run",
            "--branch",
            f"--data-file={cov_data}",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-o",
            "pythonpath=",
            "-q",
        ]
        + suite_files,
        cwd=root,
        env=env,
        timeout=timeout,
    )

    if cov_data.exists():
        _ = run_cmd(
            [
                coverage_python,
                "-m",
                "coverage",
                "json",
                f"--data-file={cov_data}",
                "-o",
                str(cov_json),
            ],
            cwd=root,
            env=env,
            timeout=timeout,
        )

    line_tokens, branch_tokens = parse_coverage_json(cov_json, root, source_prefix)
    return {
        "status": "ok" if run["returncode"] == 0 else "failed",
        "runtime_ms": round(run["runtime_ms"], 3),
        "line_tokens": line_tokens,
        "branch_tokens": branch_tokens,
        "error": "" if run["returncode"] == 0 else run["output"][:1200],
    }


def write_coverage_artifacts(
    root: Path,
    out_dir: Path,
    tests: list[TestMeta],
    ranked_map: dict[str, dict[str, str]],
    ranked_path: Path | None,
    comparator_suite_files: list[str],
    *,
    python_exe: str,
    env: dict[str, str],
    timeout: int,
    max_workers: int,
    source_prefix: str = "",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cov_keys = {
        "runtime_ms",
        "executed_line_count",
        "executed_branch_count",
        "unique_line_count",
        "unique_branch_count",
        "cross_suite_overlap_ratio",
    }
    headers = [
        "test_nodeid",
        "file",
        "entrypoint",
        "intent",
        "coverage_signal_available",
        "status",
        "runtime_ms",
        "executed_line_count",
        "executed_branch_count",
        "unique_line_count",
        "unique_branch_count",
        "cross_suite_overlap_ratio",
        "source_ranked_csv",
        "status_note",
        "error",
    ]

    has_ranked_cov = any(any(k in row for k in cov_keys) for row in ranked_map.values())
    if has_ranked_cov:
        rows: list[dict[str, Any]] = []
        available = 0
        for t in tests:
            ranked = ranked_map.get(t.nodeid, {})
            has_cov_cols = any(k in ranked for k in cov_keys)
            if has_cov_cols:
                available += 1
                note = "coverage signal loaded from ranked_report.csv"
            else:
                note = "ranked report present, but test_nodeid has no coverage row"
            rows.append(
                {
                    "test_nodeid": t.nodeid,
                    "file": t.file,
                    "entrypoint": t.entrypoint,
                    "intent": t.intent,
                    "coverage_signal_available": has_cov_cols,
                    "status": "from_ranked",
                    "runtime_ms": ranked.get("runtime_ms", ""),
                    "executed_line_count": ranked.get("executed_line_count", ""),
                    "executed_branch_count": ranked.get("executed_branch_count", ""),
                    "unique_line_count": ranked.get("unique_line_count", ""),
                    "unique_branch_count": ranked.get("unique_branch_count", ""),
                    "cross_suite_overlap_ratio": ranked.get("cross_suite_overlap_ratio", ""),
                    "source_ranked_csv": str(ranked_path) if ranked_path else "",
                    "status_note": note,
                    "error": "",
                    "_line_tokens": set(),
                    "_branch_tokens": set(),
                }
            )
        write_csv(out_dir / "coverage_matrix.csv", rows, headers)
        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": "ranked_report",
            "tests_total": len(tests),
            "with_coverage_signal": available,
            "without_coverage_signal": len(tests) - available,
            "comparator_status": "from_ranked",
            "coverage_python": "",
        }
        (out_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {r["test_nodeid"]: r for r in rows}, summary

    coverage_python, coverage_env, coverage_mode, coverage_note = ensure_coverage_tool(
        root,
        out_dir,
        python_exe,
        env,
        timeout,
    )
    if not coverage_python:
        rows = []
        for t in tests:
            rows.append(
                {
                    "test_nodeid": t.nodeid,
                    "file": t.file,
                    "entrypoint": t.entrypoint,
                    "intent": t.intent,
                    "coverage_signal_available": False,
                    "status": "unavailable",
                    "runtime_ms": "",
                    "executed_line_count": "",
                    "executed_branch_count": "",
                    "unique_line_count": "",
                    "unique_branch_count": "",
                    "cross_suite_overlap_ratio": "",
                    "source_ranked_csv": "",
                    "status_note": coverage_note,
                    "error": "coverage unavailable",
                    "_line_tokens": set(),
                    "_branch_tokens": set(),
                }
            )
        write_csv(out_dir / "coverage_matrix.csv", rows, headers)
        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": "unavailable",
            "tests_total": len(tests),
            "with_coverage_signal": 0,
            "without_coverage_signal": len(tests),
            "comparator_status": "not_run",
            "coverage_python": "",
            "status_note": coverage_note,
        }
        (out_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {r["test_nodeid"]: r for r in rows}, summary

    nodeids = [t.nodeid for t in tests]
    with tempfile.TemporaryDirectory(prefix="triage_cov_") as td:
        tmp_dir = Path(td)
        comparator = collect_suite_coverage_union(
            root,
            comparator_suite_files,
            coverage_python=coverage_python,
            env=coverage_env,
            timeout=timeout,
            tmp_dir=tmp_dir,
            source_prefix=source_prefix,
        )
        comparator_union: set[str] = set()
        if comparator["status"] == "ok":
            comparator_union = comparator["line_tokens"] | comparator["branch_tokens"]

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futs = [
                ex.submit(
                    run_single_test_coverage,
                    root,
                    nodeid,
                    coverage_python=coverage_python,
                    env=coverage_env,
                    timeout=timeout,
                    tmp_dir=tmp_dir,
                    source_prefix=source_prefix,
                )
                for nodeid in nodeids
            ]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append({"test_nodeid": "", "status": "error", "error": str(exc)})

    by_nodeid = {r["test_nodeid"]: r for r in results if r.get("test_nodeid")}
    rows = []
    available = 0
    t_by_nodeid = {t.nodeid: t for t in tests}
    for nodeid in nodeids:
        run_row = by_nodeid.get(
            nodeid,
            {
                "status": "failed",
                "runtime_ms": 0.0,
                "executed_line_count": 0,
                "executed_branch_count": 0,
                "_line_tokens": set(),
                "_branch_tokens": set(),
                "error": "missing coverage row",
            },
        )
        line_set = set(run_row.get("_line_tokens", set()))
        branch_set = set(run_row.get("_branch_tokens", set()))

        other_lines: set[str] = set()
        other_branches: set[str] = set()
        for other in nodeids:
            if other == nodeid:
                continue
            o = by_nodeid.get(other, {})
            other_lines |= set(o.get("_line_tokens", set()))
            other_branches |= set(o.get("_branch_tokens", set()))

        unique_lines = line_set - other_lines
        unique_branches = branch_set - other_branches

        overlap_value: float | str = ""
        status_note = "coverage collected without comparator suite; overlap unavailable"
        if comparator["status"] == "ok":
            full_set = line_set | branch_set
            overlap_value = round(len(full_set & comparator_union) / len(full_set), 6) if full_set else 1.0
            status_note = "coverage collected with comparator overlap"
        elif comparator["status"] == "failed":
            status_note = "coverage collected; comparator suite failed so overlap unavailable"

        signal_available = run_row.get("status") in {"passed", "skipped"}
        if signal_available:
            available += 1
        t = t_by_nodeid[nodeid]
        rows.append(
            {
                "test_nodeid": nodeid,
                "file": t.file,
                "entrypoint": t.entrypoint,
                "intent": t.intent,
                "coverage_signal_available": signal_available,
                "status": run_row.get("status", "failed"),
                "runtime_ms": run_row.get("runtime_ms", ""),
                "executed_line_count": run_row.get("executed_line_count", 0),
                "executed_branch_count": run_row.get("executed_branch_count", 0),
                "unique_line_count": len(unique_lines),
                "unique_branch_count": len(unique_branches),
                "cross_suite_overlap_ratio": overlap_value,
                "source_ranked_csv": "",
                "status_note": status_note,
                "error": run_row.get("error", ""),
                "_line_tokens": line_set,
                "_branch_tokens": branch_set,
            }
        )

    write_csv(out_dir / "coverage_matrix.csv", rows, headers)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": coverage_mode,
        "tests_total": len(tests),
        "with_coverage_signal": available,
        "without_coverage_signal": len(tests) - available,
        "coverage_python": coverage_python,
        "coverage_note": coverage_note,
        "comparator_status": comparator["status"],
        "comparator_runtime_ms": comparator["runtime_ms"],
        "comparator_error": comparator.get("error", ""),
    }
    (out_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {r["test_nodeid"]: r for r in rows}, summary


def write_inventory_artifact(out_dir: Path, tests: list[TestMeta]) -> None:
    rows: list[dict[str, Any]] = []
    for t in tests:
        rows.append(
            {
                "test_nodeid": t.nodeid,
                "file": t.file,
                "class_name": t.class_name,
                "test_name": t.test_name,
                "entrypoint": t.entrypoint,
                "intent": t.intent,
                "assertion_types": ";".join(sorted(t.assertion_types)),
                "assert_count": t.assert_count,
                "is_parametrized": t.is_parametrized,
            }
        )
    headers = [
        "test_nodeid",
        "file",
        "class_name",
        "test_name",
        "entrypoint",
        "intent",
        "assertion_types",
        "assert_count",
        "is_parametrized",
    ]
    write_csv(out_dir / "inventory.csv", rows, headers)


def write_mutation_artifacts(
    out_dir: Path,
    tests: list[TestMeta],
    ranked_map: dict[str, dict[str, str]],
    ranked_path: Path | None,
) -> None:
    """Always emit mutation artifacts, even when ranked mutation signal is absent.

    If ranked rows include mutation columns, values are propagated.
    Otherwise rows are emitted with availability flags and an explanatory note.
    """
    rows: list[dict[str, Any]] = []
    available = 0
    for t in tests:
        ranked = ranked_map.get(t.nodeid, {})
        has_mut_cols = bool(ranked) and any(
            k in ranked for k in ("mutants_unique_to_api", "mutants_killed_api", "mutants_killed_non_api")
        )
        if has_mut_cols:
            available += 1
            note = "mutation signal loaded from ranked_report.csv"
        elif ranked_map:
            note = "ranked report present, but test_nodeid has no mutation row"
        else:
            note = "ranked report missing; mutation signal unavailable for this run"
        rows.append(
            {
                "test_nodeid": t.nodeid,
                "file": t.file,
                "entrypoint": t.entrypoint,
                "intent": t.intent,
                "mutation_signal_available": has_mut_cols,
                "mutants_killed_primary": ranked.get("mutants_killed_api", ""),
                "mutants_killed_secondary": ranked.get("mutants_killed_non_api", ""),
                "mutants_unique_primary": ranked.get("mutants_unique_to_api", ""),
                "source_ranked_csv": str(ranked_path) if ranked_path else "",
                "status_note": note,
            }
        )

    headers = [
        "test_nodeid",
        "file",
        "entrypoint",
        "intent",
        "mutation_signal_available",
        "mutants_killed_primary",
        "mutants_killed_secondary",
        "mutants_unique_primary",
        "source_ranked_csv",
        "status_note",
    ]
    write_csv(out_dir / "mutation_matrix.csv", rows, headers)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tests_total": len(tests),
        "with_mutation_signal": available,
        "without_mutation_signal": len(tests) - available,
        "ranked_csv_provided": bool(ranked_path),
        "ranked_csv_exists": bool(ranked_path and ranked_path.exists()),
    }
    (out_dir / "mutation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    low = str(val).strip().lower()
    return low in {"1", "true", "yes", "y"}


def select_branch_anchor(
    candidate: TestMeta,
    peers: list[TestMeta],
    decision_by_nodeid: dict[str, str],
) -> TestMeta | None:
    if not peers:
        return None

    # Prefer peers that are currently retained (non-delete decisions).
    keep_peers = [p for p in peers if decision_by_nodeid.get(p.nodeid, "") != "DELETE_SAFE_HIGH"]
    pool = keep_peers if keep_peers else peers

    ranked = sorted(
        pool,
        key=lambda p: (
            jaccard_sim(candidate.src_tokens, p.src_tokens),
            difflib.SequenceMatcher(None, candidate.test_name, p.test_name).ratio(),
            len(candidate.assertion_types & p.assertion_types),
            p.assert_count,
            p.nodeid,
        ),
    )
    return ranked[-1] if ranked else None


def collect_node_coverage_runs(
    root: Path,
    out_dir: Path,
    nodeids: list[str],
    *,
    python_exe: str,
    env: dict[str, str],
    timeout: int,
    max_workers: int,
    source_prefix: str = "",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not nodeids:
        return {}, {
            "mode": "not_needed",
            "status_note": "all nodeids already had coverage token sets",
            "tests_total": 0,
            "passed_or_skipped": 0,
            "failed": 0,
        }

    coverage_python, coverage_env, coverage_mode, coverage_note = ensure_coverage_tool(
        root,
        out_dir,
        python_exe,
        env,
        timeout,
    )
    if not coverage_python:
        return {}, {
            "mode": "unavailable",
            "status_note": coverage_note,
            "tests_total": len(nodeids),
            "passed_or_skipped": 0,
            "failed": len(nodeids),
            "coverage_python": "",
        }

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="triage_branch_cov_") as td:
        tmp_dir = Path(td)
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futs = [
                ex.submit(
                    run_single_test_coverage,
                    root,
                    nodeid,
                    coverage_python=coverage_python,
                    env=coverage_env,
                    timeout=timeout,
                    tmp_dir=tmp_dir,
                    source_prefix=source_prefix,
                )
                for nodeid in nodeids
            ]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append({"test_nodeid": "", "status": "error", "error": str(exc)})

    passed_or_skipped = sum(1 for r in results if r.get("status") in {"passed", "skipped"})
    summary = {
        "mode": coverage_mode,
        "status_note": coverage_note,
        "tests_total": len(nodeids),
        "passed_or_skipped": passed_or_skipped,
        "failed": len(nodeids) - passed_or_skipped,
        "coverage_python": coverage_python,
    }
    return {r["test_nodeid"]: r for r in results}, summary


def write_branch_equiv_artifacts(
    root: Path,
    out_dir: Path,
    tests: list[TestMeta],
    by_cluster: dict[tuple[str, str], list[TestMeta]],
    rows: list[dict[str, Any]],
    coverage_map: dict[str, dict[str, Any]],
    *,
    python_exe: str,
    env: dict[str, str],
    timeout: int,
    max_workers: int,
    source_prefix: str = "",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    csv_path = out_dir / "branch_equiv_report.csv"
    json_path = out_dir / "branch_equiv_summary.json"
    md_path = out_dir / "branch_equiv_report.md"
    headers = [
        "candidate",
        "anchor",
        "candidate_decision",
        "anchor_decision",
        "candidate_status",
        "anchor_status",
        "candidate_branch_count",
        "anchor_branch_count",
        "branch_jaccard",
        "exact_branch_match",
        "candidate_only_branches",
        "anchor_only_branches",
        "candidate_runtime_ms",
        "anchor_runtime_ms",
        "sample_candidate_only",
        "sample_anchor_only",
    ]

    test_by_nodeid = {t.nodeid: t for t in tests}
    decision_by_nodeid = {str(r.get("test_nodeid", "")): str(r.get("validation_decision", "")) for r in rows}
    pair_defs: list[tuple[str, str]] = []
    for r in rows:
        nodeid = str(r.get("test_nodeid", ""))
        if not nodeid or nodeid not in test_by_nodeid:
            continue
        if not as_bool(r.get("deselect_suite_pass", False)):
            continue
        decision = str(r.get("validation_decision", ""))
        if decision not in {"DELETE_SAFE_HIGH", "MERGE_RECOMMENDED", "KEEP_FOR_SIGNAL", "KEEP_FOR_CONTRACT"}:
            continue
        candidate = test_by_nodeid[nodeid]
        peers = [p for p in by_cluster[(candidate.entrypoint, candidate.intent)] if p.nodeid != nodeid]
        anchor = select_branch_anchor(candidate, peers, decision_by_nodeid)
        if anchor:
            pair_defs.append((nodeid, anchor.nodeid))

    pair_defs = unique_preserve([f"{c}|||{a}" for c, a in pair_defs])
    resolved_pairs = [(p.split("|||", 1)[0], p.split("|||", 1)[1]) for p in pair_defs]

    branch_cache: dict[str, dict[str, Any]] = {}
    for nodeid, cov_row in coverage_map.items():
        line_tokens = cov_row.get("_line_tokens")
        branch_tokens = cov_row.get("_branch_tokens")
        if isinstance(line_tokens, set) and isinstance(branch_tokens, set):
            branch_cache[nodeid] = {
                "status": cov_row.get("status", ""),
                "runtime_ms": cov_row.get("runtime_ms", ""),
                "_line_tokens": set(line_tokens),
                "_branch_tokens": set(branch_tokens),
            }

    def _needs_live_collection(nodeid: str) -> bool:
        """Return True when the branch cache entry is absent or a ranked-mode placeholder."""
        cached = branch_cache.get(nodeid)
        return cached is None or cached.get("status") == "from_ranked"

    needed_nodeids: list[str] = []
    for cand, anchor in resolved_pairs:
        if _needs_live_collection(cand):
            needed_nodeids.append(cand)
        if _needs_live_collection(anchor):
            needed_nodeids.append(anchor)
    needed_nodeids = unique_preserve(needed_nodeids)

    extra_cov, extra_cov_summary = collect_node_coverage_runs(
        root,
        out_dir,
        needed_nodeids,
        python_exe=python_exe,
        env=env,
        timeout=timeout,
        max_workers=max_workers,
        source_prefix=source_prefix,
    )
    for nodeid, cov in extra_cov.items():
        branch_cache[nodeid] = cov

    report_rows: list[dict[str, Any]] = []
    branch_result_by_candidate: dict[str, dict[str, Any]] = {}
    exact = 0
    non_exact = 0
    all_tests_passed: bool | None = None if not resolved_pairs else True
    for candidate_nodeid, anchor_nodeid in resolved_pairs:
        candidate_cov = branch_cache.get(candidate_nodeid, {})
        anchor_cov = branch_cache.get(anchor_nodeid, {})
        candidate_status = str(candidate_cov.get("status", "missing"))
        anchor_status = str(anchor_cov.get("status", "missing"))
        candidate_branches = set(candidate_cov.get("_branch_tokens", set()))
        anchor_branches = set(anchor_cov.get("_branch_tokens", set()))

        comparable = (
            candidate_status in {"passed", "skipped"}
            and anchor_status in {"passed", "skipped"}
        )
        if candidate_status not in {"passed", "skipped"} or anchor_status not in {"passed", "skipped"}:
            all_tests_passed = False

        candidate_only = candidate_branches - anchor_branches
        anchor_only = anchor_branches - candidate_branches
        union = candidate_branches | anchor_branches
        branch_jaccard: float | str = ""
        exact_branch_match: bool | str = ""
        if comparable:
            branch_jaccard = round(len(candidate_branches & anchor_branches) / len(union), 6) if union else 1.0
            exact_branch_match = candidate_branches == anchor_branches
            if exact_branch_match:
                exact += 1
            else:
                non_exact += 1

        report = {
            "candidate": candidate_nodeid,
            "anchor": anchor_nodeid,
            "candidate_decision": decision_by_nodeid.get(candidate_nodeid, ""),
            "anchor_decision": decision_by_nodeid.get(anchor_nodeid, ""),
            "candidate_status": candidate_status,
            "anchor_status": anchor_status,
            "candidate_branch_count": len(candidate_branches),
            "anchor_branch_count": len(anchor_branches),
            "branch_jaccard": branch_jaccard,
            "exact_branch_match": exact_branch_match,
            "candidate_only_branches": len(candidate_only),
            "anchor_only_branches": len(anchor_only),
            "candidate_runtime_ms": candidate_cov.get("runtime_ms", ""),
            "anchor_runtime_ms": anchor_cov.get("runtime_ms", ""),
            "sample_candidate_only": " | ".join(sorted(candidate_only)[:5]),
            "sample_anchor_only": " | ".join(sorted(anchor_only)[:5]),
        }
        report_rows.append(report)
        branch_result_by_candidate[candidate_nodeid] = {
            "branch_anchor_nodeid": anchor_nodeid,
            "branch_exact_match": exact_branch_match,
            "branch_jaccard": branch_jaccard,
            "branch_candidate_only_count": len(candidate_only),
            "branch_anchor_only_count": len(anchor_only),
            "branch_status_note": (
                "branch equivalence unavailable (coverage status not passed/skipped)"
                if exact_branch_match == ""
                else "branch equivalence computed"
            ),
        }

    write_csv(csv_path, report_rows, headers)
    mode = "coverage_matrix_tokens"
    if not resolved_pairs:
        mode = "no_pairs"
    if needed_nodeids:
        mode = "coverage_matrix_tokens+live_collection"
    if needed_nodeids and not extra_cov:
        mode = "coverage_unavailable_for_missing"
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pairs_total": len(report_rows),
        "exact_branch_matches": exact,
        "non_exact_branch_matches": non_exact,
        "all_tests_passed": all_tests_passed,
        "coverage_mode": mode,
        "coverage_note": extra_cov_summary.get("status_note", ""),
        "live_collection_tests": len(needed_nodeids),
        "live_collection_summary": extra_cov_summary,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md: list[str] = [
        "# Branch Equivalence Report",
        "",
        "## Summary",
        "",
        f"- `generated_at`: `{summary['generated_at']}`",
        f"- `coverage_mode`: `{summary['coverage_mode']}`",
        f"- `coverage_note`: `{summary['coverage_note']}`",
        f"- `pairs_total`: `{summary['pairs_total']}`",
        f"- `exact_branch_matches`: `{summary['exact_branch_matches']}`",
        f"- `non_exact_branch_matches`: `{summary['non_exact_branch_matches']}`",
        f"- `all_tests_passed`: `{summary['all_tests_passed']}`",
        "",
        "## Pair Results",
        "",
        "| candidate | anchor | exact_branch_match | jaccard | cand_only | anchor_only |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(report_rows, key=lambda rr: (str(rr.get("branch_jaccard", "")), str(rr.get("candidate", "")))):
        md.append(
            "| "
            f"{r.get('candidate','')} | {r.get('anchor','')} | {r.get('exact_branch_match','')} | "
            f"{r.get('branch_jaccard','')} | {r.get('candidate_only_branches','')} | {r.get('anchor_only_branches','')} |"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    return branch_result_by_candidate, summary


def bool_low_signal(
    meta: TestMeta,
    ranked: dict[str, str],
    coverage_row: dict[str, Any] | None,
    *,
    branch_equiv_row: dict | None = None,
) -> bool:
    # If branch-equivalence data shows this test has unique branches, it is NOT low-signal.
    if branch_equiv_row is not None:
        cand_only = as_int(branch_equiv_row.get("branch_candidate_only_count", 0))
        if cand_only > 0:
            return False
    if ranked:
        ul = int(ranked.get("unique_line_count", 0) or 0)
        ub = int(ranked.get("unique_branch_count", 0) or 0)
        mu = int(ranked.get("mutants_unique_to_api", 0) or 0)
        ov = float(ranked.get("cross_suite_overlap_ratio", 1.0) or 1.0)
        return ul == 0 and ub == 0 and mu == 0 and ov >= 0.97

    if coverage_row and str(coverage_row.get("coverage_signal_available", "")).lower() in {"true", "1"}:
        ov_raw = coverage_row.get("cross_suite_overlap_ratio", "")
        if ov_raw in {"", None}:
            return False
        ul = as_int(coverage_row.get("unique_line_count", 0))
        ub = as_int(coverage_row.get("unique_branch_count", 0))
        ov = as_float(ov_raw)
        if ul == 0 and ub == 0 and ov >= 0.97:
            complex_signals = {"exception", "mutability_contract", "dtype_contract", "array_equality", "topology_equality"}
            return len(meta.assertion_types & complex_signals) == 0

    # AST-only fallback, intentionally conservative.
    # Empty assertion_types means assertions couldn't be classified — treat as
    # unknown signal rather than low signal to avoid false deletes.
    if not meta.assertion_types:
        return False
    smoke = {"type_check", "length_contract", "general_assert"}
    return meta.assertion_types.issubset(smoke)


def as_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def as_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def as_bool_any(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


def tri_state(val: bool | None) -> str:
    if val is None:
        return "unknown"
    return "pass" if val else "fail"


def write_confidence_gate_artifact(
    out_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Emit hardened confidence gates and tier for each evaluated row.

    This is intentionally generic and repository-agnostic:
    - dynamic behavior gates come from deselection/strict/branch signal
    - semantic gates come from assertion surfaces and dominance
    - mutation/coverage gates come from uniqueness metrics
    - external gates are explicitly marked unknown for manual signoff
    """
    gate_headers = [
        "test_nodeid",
        "validation_decision",
        "gate_deselect",
        "gate_strict_stability",
        "gate_branch_exact",
        "gate_branch_high_similarity",
        "gate_assertion_dominance",
        "gate_unique_mutants",
        "gate_unique_coverage",
        "gate_cross_suite_overlap",
        "gate_external_side_effects",
        "gate_external_nonfunctional",
        "gate_external_history_regression",
        "confidence_tier",
        "manual_signoff_needed",
        "confidence_note",
    ]

    tier_counts: dict[str, int] = defaultdict(int)
    gate_rows: list[dict[str, Any]] = []
    for r in rows:
        nodeid = str(r.get("test_nodeid", ""))
        decision = str(r.get("validation_decision", ""))

        if decision == "BASELINE_FAILED":
            tier = "BASELINE_FAILED"
            note = "baseline failed; no candidate evidence available"
            gate_row = {
                "test_nodeid": nodeid,
                "validation_decision": decision,
                "gate_deselect": "fail",
                "gate_strict_stability": "unknown",
                "gate_branch_exact": "unknown",
                "gate_branch_high_similarity": "unknown",
                "gate_assertion_dominance": "unknown",
                "gate_unique_mutants": "unknown",
                "gate_unique_coverage": "unknown",
                "gate_cross_suite_overlap": "unknown",
                "gate_external_side_effects": "unknown",
                "gate_external_nonfunctional": "unknown",
                "gate_external_history_regression": "unknown",
                "confidence_tier": tier,
                "manual_signoff_needed": True,
                "confidence_note": note,
            }
            gate_rows.append(gate_row)
            r.update(gate_row)
            tier_counts[tier] += 1
            continue

        deselect_pass = as_bool_any(r.get("deselect_suite_pass", False))
        strict_status = str(r.get("strict_gate_status", "") or "")
        strict_gate: bool | None = None
        if strict_status == "passed":
            strict_gate = True
        elif strict_status in {"failed", "not_run", "not_processed"}:
            strict_gate = False
        elif strict_status in {"disabled", "not_applicable", "pending", ""}:
            strict_gate = None

        branch_exact_raw = str(r.get("branch_exact_match", "")).strip().lower()
        if branch_exact_raw in {"true", "false"}:
            branch_exact: bool | None = branch_exact_raw == "true"
        else:
            branch_exact = None

        branch_jaccard_raw = r.get("branch_jaccard", "")
        branch_jaccard: float | None = None
        if branch_jaccard_raw not in {"", None}:
            try:
                branch_jaccard = float(branch_jaccard_raw)
            except (TypeError, ValueError):
                branch_jaccard = None
        branch_high: bool | None = None if branch_jaccard is None else branch_jaccard >= 0.95

        dominated = as_bool_any(r.get("peer_superset_assertions", False))

        mut_unique_raw = r.get("report_mutants_unique_to_api", "")
        mut_unique: bool | None = None
        if mut_unique_raw not in {"", None}:
            mut_unique = as_int(mut_unique_raw) == 0

        ul_raw = r.get("report_unique_line_count", "")
        ub_raw = r.get("report_unique_branch_count", "")
        cov_unique: bool | None = None
        if ul_raw not in {"", None} and ub_raw not in {"", None}:
            cov_unique = as_int(ul_raw) == 0 and as_int(ub_raw) == 0

        overlap_raw = r.get("report_overlap", "")
        overlap_high: bool | None = None
        if overlap_raw not in {"", None}:
            overlap_high = as_float(overlap_raw) >= 0.97

        measured_strong = (
            deselect_pass
            and dominated
            and (mut_unique is True)
            and (cov_unique is True)
            and (overlap_high is True)
            and (branch_exact is True)
            and (strict_gate is True)
        )
        measured_good = (
            deselect_pass
            and dominated
            and (mut_unique is True)
            and (cov_unique is True)
            and (overlap_high is True)
            and ((branch_exact is True) or (branch_high is True))
            and (strict_gate in {True, None})
        )

        if decision == "DELETE_SAFE_HIGH":
            if measured_strong:
                tier = "GOLD_DELETE_CANDIDATE"
                note = "all measured gates passed with exact branch-equivalence"
            elif measured_good:
                tier = "SILVER_DELETE_CANDIDATE"
                note = "all measured core gates passed; branch gate is high-similarity or strict gate unavailable"
            else:
                tier = "BRONZE_DELETE_REVIEW"
                note = "delete candidate failed one or more measured gates"
        elif decision == "MERGE_RECOMMENDED":
            tier = "MERGE_CANDIDATE"
            note = "overlap strong but delete gates not fully satisfied"
        elif decision in {"KEEP_FOR_SIGNAL", "KEEP_FOR_CONTRACT", "KEEP_FOR_STABILITY"}:
            tier = "KEEP_CANDIDATE"
            note = "retain test based on stability, signal, or contract safeguards"
        else:
            tier = "UNCLASSIFIED"
            note = "decision outside standard confidence model"

        manual_signoff = tier in {"SILVER_DELETE_CANDIDATE", "BRONZE_DELETE_REVIEW"}

        gate_row = {
            "test_nodeid": nodeid,
            "validation_decision": decision,
            "gate_deselect": tri_state(deselect_pass),
            "gate_strict_stability": tri_state(strict_gate),
            "gate_branch_exact": tri_state(branch_exact),
            "gate_branch_high_similarity": tri_state(branch_high),
            "gate_assertion_dominance": tri_state(dominated),
            "gate_unique_mutants": tri_state(mut_unique),
            "gate_unique_coverage": tri_state(cov_unique),
            "gate_cross_suite_overlap": tri_state(overlap_high),
            "gate_external_side_effects": "unknown",
            "gate_external_nonfunctional": "unknown",
            "gate_external_history_regression": "unknown",
            "confidence_tier": tier,
            "manual_signoff_needed": manual_signoff,
            "confidence_note": note,
        }
        gate_rows.append(gate_row)
        r.update(gate_row)
        tier_counts[tier] += 1

    write_csv(out_dir / "confidence_gate_matrix.csv", gate_rows, gate_headers)
    return dict(tier_counts)


def enforce_cluster_anchor(rows: list[dict[str, Any]]) -> None:
    """Prevent an accidental full-cluster prune in one deletion batch.

    If every candidate in a cluster is tagged DELETE_SAFE_HIGH, retain one
    anchor as KEEP_FOR_CONTRACT based on strongest available signal.
    """
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        entry = str(r.get("entrypoint", "") or "")
        intent = str(r.get("intent", "") or "")
        if entry and intent:
            clusters[(entry, intent)].append(r)

    for cluster_rows in clusters.values():
        if not cluster_rows:
            continue
        if not all(r.get("validation_decision") == "DELETE_SAFE_HIGH" for r in cluster_rows):
            continue

        anchor = max(
            cluster_rows,
            key=lambda r: (
                as_int(r.get("report_unique_line_count")),
                as_int(r.get("report_unique_branch_count")),
                as_int(r.get("report_mutants_unique_to_api")),
                as_int(r.get("assert_count")),
                -as_float(r.get("max_src_similarity")),
            ),
        )
        anchor["validation_decision"] = "KEEP_FOR_CONTRACT"
        anchor["validation_reason"] = (
            "cluster anchor retained; deleting all flagged tests would empty this entrypoint+intent cluster"
        )


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items[:]] if items else []
    return [items[i : i + size] for i in range(0, len(items), size)]


def unique_preserve(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_default_mutation_probes() -> list[MutationProbe]:
    # No built-in probes are provided; supply probes via --mutation-probes-config.
    # See SKILL.md for the JSON format.
    return []


def load_mutation_probes_from_config(config_path: Path) -> list[MutationProbe]:
    """Load mutation probes from a JSON config file.

    Expected format: list of objects with keys probe_id, file, old, new.
    """
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to load mutation probes config {config_path}: {exc}") from exc
    probes: list[MutationProbe] = []
    for i, entry in enumerate(raw):
        try:
            probes.append(MutationProbe(
                probe_id=entry["probe_id"],
                file=entry["file"],
                old=entry["old"],
                new=entry["new"],
            ))
        except KeyError as exc:
            raise SystemExit(f"Mutation probes config entry {i} missing required key {exc}") from exc
    return probes


def apply_mutation_probe(overlay_root: Path, probe: MutationProbe) -> tuple[bool, str]:
    target = overlay_root / probe.file
    if not target.exists():
        return False, f"target file missing for probe {probe.probe_id}: {probe.file}"
    text = target.read_text(encoding="utf-8")
    count = text.count(probe.old)
    if count != 1:
        return False, f"probe {probe.probe_id} replacement count mismatch: expected 1, got {count}"
    target.write_text(text.replace(probe.old, probe.new, 1), encoding="utf-8")
    return True, ""


def build_overlay_env(root: Path, env: dict[str, str], overlay_root: Path) -> dict[str, str]:
    out = dict(env)
    existing = out.get("PYTHONPATH", "")
    root_import_roots = [Path(p) for p in discover_import_roots(root)]
    overlay_prefixes: list[str] = []
    for root_path in root_import_roots:
        if root_path == root:
            overlay_prefixes.append(str(overlay_root))
            continue
        try:
            rel = root_path.relative_to(root)
        except ValueError:
            continue
        overlay_candidate = overlay_root / rel
        if overlay_candidate.exists():
            overlay_prefixes.append(str(overlay_candidate))
    prefixes = unique_preserve(overlay_prefixes + [str(p) for p in root_import_roots])
    if existing:
        prefixes.append(existing)
    out["PYTHONPATH"] = os.pathsep.join(prefixes)
    return out


def stage_probe_file_overlay(root: Path, overlay_root: Path, probe: MutationProbe) -> tuple[bool, str]:
    """Copy only the mutated file (and package __init__.py chain) into overlay."""
    src_file = (root / probe.file).resolve()
    try:
        src_file.relative_to(root)
    except ValueError:
        return False, f"probe {probe.probe_id} target must be inside --root: {probe.file}"
    if not src_file.exists():
        return False, f"target file missing for probe {probe.probe_id}: {probe.file}"

    rel_file = src_file.relative_to(root)
    dst_file = overlay_root / rel_file
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)

    # Preserve package semantics for imports by copying existing __init__.py chain.
    cur = src_file.parent
    while True:
        init_py = cur / "__init__.py"
        if init_py.exists():
            rel_init = init_py.relative_to(root)
            dst_init = overlay_root / rel_init
            dst_init.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(init_py, dst_init)
        if cur == root:
            break
        cur = cur.parent
        try:
            cur.relative_to(root)
        except ValueError:
            break
    return True, ""


def run_mutation_probe_kills(
    root: Path,
    python_exe: str,
    env: dict[str, str],
    suite_files: list[str],
    deselects: list[str],
    timeout: int,
    *,
    use_xdist: bool,
    probes: list[MutationProbe],
) -> dict[str, Any]:
    if not probes:
        return {
            "status": "skipped",
            "kills": 0,
            "total_probes": 0,
            "applied_probes": 0,
            "failed_to_apply": 0,
            "details": [],
            "error": "",
        }

    details: list[dict[str, Any]] = []
    kills = 0
    applied_count = 0
    for probe in probes:
        with tempfile.TemporaryDirectory(prefix=f"triage_probe_{probe.probe_id}_") as td:
            overlay_root = Path(td)
            staged, stage_error = stage_probe_file_overlay(root, overlay_root, probe)
            if not staged:
                details.append(
                    {
                        "probe_id": probe.probe_id,
                        "killed": False,
                        "applied": False,
                        "runtime_ms": 0.0,
                        "error": stage_error,
                    }
                )
                continue
            applied, error = apply_mutation_probe(overlay_root, probe)
            if not applied:
                details.append(
                    {
                        "probe_id": probe.probe_id,
                        "killed": False,
                        "applied": False,
                        "runtime_ms": 0.0,
                        "error": error,
                    }
                )
                continue

            applied_count += 1
            overlay_env = build_overlay_env(root, env, overlay_root)
            run = run_suite_multi(
                root,
                python_exe,
                suite_files,
                overlay_env,
                deselects,
                timeout,
                use_xdist=use_xdist,
            )
            killed = run["returncode"] != 0
            kills += int(killed)
            details.append(
                {
                    "probe_id": probe.probe_id,
                    "killed": killed,
                    "applied": True,
                    "runtime_ms": round(run["runtime_ms"], 3),
                    "error": "" if killed else run["output"][:300],
                }
            )

    failed_count = max(0, len(probes) - applied_count)
    failed_ids = ", ".join(d["probe_id"] for d in details if not d.get("applied", False))

    if applied_count == 0:
        return {
            "status": "no_probes_applied",
            "kills": 0,
            "total_probes": len(probes),
            "applied_probes": 0,
            "failed_to_apply": len(probes),
            "details": details,
            "error": f"all {len(probes)} probe(s) failed to apply ({failed_ids}); mutation gate is not exercised",
        }

    if applied_count < len(probes):
        return {
            "status": "partial_probes_applied",
            "kills": kills,
            "total_probes": len(probes),
            "applied_probes": applied_count,
            "failed_to_apply": failed_count,
            "details": details,
            "error": (
                f"{failed_count}/{len(probes)} probe(s) failed to apply "
                f"({failed_ids}); strict mutation gate requires all selected probes to apply"
            ),
        }

    return {
        "status": "ok",
        "kills": kills,
        "total_probes": len(probes),
        "applied_probes": applied_count,
        "failed_to_apply": 0,
        "details": details,
        "error": "",
    }


def run_strict_delete_gate(
    root: Path,
    out_dir: Path,
    python_exe: str,
    env: dict[str, str],
    suite_files: list[str],
    post_suite_files: list[str],
    timeout: int,
    *,
    use_xdist: bool,
    rows: list[dict[str, Any]],
    repeats: int,
    batch_size: int,
    max_batches: int,
    probes_source: list[MutationProbe],
    mutation_probe_count: int,
    mutation_max_drop: int,
) -> dict[str, Any]:
    delete_rows = [r for r in rows if r.get("validation_decision") == "DELETE_SAFE_HIGH" and r.get("test_nodeid")]
    node_status: dict[str, str] = {}
    node_note: dict[str, str] = {}
    strict_rows: list[dict[str, Any]] = []
    strict_headers = [
        "batch_index",
        "batch_size",
        "batch_nodeids",
        "target_suite_pass",
        "post_suite_pass",
        "repeat_pass",
        "repeat_count",
        "repeat_fail_count",
        "mutation_gate_pass",
        "mutation_baseline_status",
        "mutation_batch_status",
        "mutation_applied_probes",
        "mutation_failed_to_apply",
        "mutation_baseline_kills",
        "mutation_batch_kills",
        "reaudit_pass",
        "decision",
        "decision_note",
        "target_failure_excerpt",
        "post_failure_excerpt",
        "repeat_failure_excerpt",
    ]

    if not post_suite_files:
        if (root / "tests").exists():
            post_suite_files = ["tests"]
        else:
            post_suite_files = list(suite_files)
    post_suite_files = unique_preserve(post_suite_files)

    if not delete_rows:
        write_csv(out_dir / "strict_gate.csv", [], strict_headers)
        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "strict_mode": True,
            "delete_candidates_total": 0,
            "processed_batches": 0,
            "accepted_delete_count": 0,
            "rejected_delete_count": 0,
            "not_processed_delete_count": 0,
            "baseline_mutation_kills": "",
            "mutation_probe_count": 0,
            "post_suite_files": post_suite_files,
        }
        (out_dir / "strict_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"node_status": node_status, "node_note": node_note, "summary": summary}

    probes = probes_source[: max(0, mutation_probe_count)]
    baseline_probe = run_mutation_probe_kills(
        root,
        python_exe,
        env,
        post_suite_files,
        deselects=[],
        timeout=timeout,
        use_xdist=use_xdist,
        probes=probes,
    )
    baseline_probe_status = str(baseline_probe.get("status", "unknown"))
    baseline_probe_ok = baseline_probe_status == "ok"
    baseline_kills = int(baseline_probe.get("kills", 0) or 0) if baseline_probe_ok else 0

    accepted: list[str] = []
    ordered = sorted(delete_rows, key=lambda r: (r.get("entrypoint", ""), r.get("intent", ""), r.get("test_nodeid", "")))
    batches = chunked(ordered, max(1, batch_size))
    if max_batches > 0:
        batches = batches[:max_batches]
    processed_nodeids: set[str] = set()

    for batch_idx, batch in enumerate(batches, start=1):
        batch_nodeids = [str(r.get("test_nodeid", "")) for r in batch if r.get("test_nodeid")]
        processed_nodeids.update(batch_nodeids)
        pending = accepted + batch_nodeids

        repeat_pass = True
        repeat_fail_count = 0
        repeat_fail_excerpt = ""
        for _ in range(max(1, repeats)):
            rr = run_suite_multi(
                root,
                python_exe,
                suite_files,
                env,
                pending,
                timeout,
                use_xdist=use_xdist,
            )
            if rr["returncode"] != 0:
                repeat_pass = False
                repeat_fail_count += 1
                if not repeat_fail_excerpt:
                    repeat_fail_excerpt = rr["output"][:300]
                break

        target_run = run_suite_multi(
            root,
            python_exe,
            suite_files,
            env,
            pending,
            timeout,
            use_xdist=use_xdist,
        )
        target_pass = target_run["returncode"] == 0

        post_run = run_suite_multi(
            root,
            python_exe,
            post_suite_files,
            env,
            pending,
            timeout,
            use_xdist=use_xdist,
        )
        post_pass = post_run["returncode"] == 0

        mutation_pass = True
        batch_kills = baseline_kills
        mutation_note = ""
        mutation_batch_status = "not_run"
        mutation_applied = ""
        mutation_failed_apply = ""
        if probes:
            if not baseline_probe_ok:
                mutation_pass = False
                mutation_note = str(baseline_probe.get("error", "") or f"baseline mutation probe run failed ({baseline_probe_status})")
                mutation_batch_status = "not_run"
                mutation_applied = str(baseline_probe.get("applied_probes", ""))
                mutation_failed_apply = str(baseline_probe.get("failed_to_apply", ""))
            else:
                batch_probe = run_mutation_probe_kills(
                    root,
                    python_exe,
                    env,
                    post_suite_files,
                    deselects=pending,
                    timeout=timeout,
                    use_xdist=use_xdist,
                    probes=probes,
                )
                mutation_batch_status = str(batch_probe.get("status", "unknown"))
                mutation_applied = str(batch_probe.get("applied_probes", ""))
                mutation_failed_apply = str(batch_probe.get("failed_to_apply", ""))
                if mutation_batch_status != "ok":
                    mutation_pass = False
                    mutation_note = str(batch_probe.get("error", "") or f"batch mutation probe run failed ({mutation_batch_status})")
                    batch_kills = 0
                else:
                    batch_kills = int(batch_probe.get("kills", 0) or 0)
                    if batch_kills < (baseline_kills - max(0, mutation_max_drop)):
                        mutation_pass = False
                        mutation_note = (
                            f"mutation kill regression: baseline={baseline_kills}, "
                            f"batch={batch_kills}, allowed_drop={max(0, mutation_max_drop)}"
                        )

        candidate_pass = repeat_pass and target_pass and post_pass and mutation_pass
        candidate_state = accepted + batch_nodeids if candidate_pass else accepted
        reaudit_run = run_suite_multi(
            root,
            python_exe,
            suite_files,
            env,
            candidate_state,
            timeout,
            use_xdist=use_xdist,
        )
        reaudit_pass = reaudit_run["returncode"] == 0

        if candidate_pass and reaudit_pass:
            accepted = candidate_state
            decision = "accepted"
            decision_note = "all strict gates passed"
            for nodeid in batch_nodeids:
                node_status[nodeid] = "passed"
                node_note[nodeid] = decision_note
        else:
            reasons = []
            if not repeat_pass:
                reasons.append(f"repeat deselect failed ({repeat_fail_count})")
            if not target_pass:
                reasons.append("target suite failed")
            if not post_pass:
                reasons.append("post suite failed")
            if not mutation_pass:
                reasons.append(mutation_note or "mutation gate failed")
            if not reaudit_pass:
                reasons.append("reaudit failed")
            decision_note = "; ".join(reasons) if reasons else "strict gate rejected batch"
            decision = "rejected"
            for nodeid in batch_nodeids:
                node_status[nodeid] = "failed"
                node_note[nodeid] = decision_note

        strict_rows.append(
            {
                "batch_index": batch_idx,
                "batch_size": len(batch_nodeids),
                "batch_nodeids": ";".join(batch_nodeids),
                "target_suite_pass": target_pass,
                "post_suite_pass": post_pass,
                "repeat_pass": repeat_pass,
                "repeat_count": max(1, repeats),
                "repeat_fail_count": repeat_fail_count,
                "mutation_gate_pass": mutation_pass,
                "mutation_baseline_status": baseline_probe_status if probes else "",
                "mutation_batch_status": mutation_batch_status if probes else "",
                "mutation_applied_probes": mutation_applied if probes else "",
                "mutation_failed_to_apply": mutation_failed_apply if probes else "",
                "mutation_baseline_kills": baseline_kills if probes and baseline_probe_ok else "",
                "mutation_batch_kills": batch_kills if probes else "",
                "reaudit_pass": reaudit_pass,
                "decision": decision,
                "decision_note": decision_note,
                "target_failure_excerpt": "" if target_pass else target_run["output"][:300],
                "post_failure_excerpt": "" if post_pass else post_run["output"][:300],
                "repeat_failure_excerpt": repeat_fail_excerpt,
            }
        )

    for r in delete_rows:
        nodeid = str(r.get("test_nodeid", ""))
        if nodeid and nodeid not in processed_nodeids and nodeid not in node_status:
            node_status[nodeid] = "not_processed"
            node_note[nodeid] = "strict gate max batch limit reached"

    write_csv(out_dir / "strict_gate.csv", strict_rows, strict_headers)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "strict_mode": True,
        "delete_candidates_total": len(delete_rows),
        "processed_batches": len(strict_rows),
        "accepted_delete_count": sum(1 for s in node_status.values() if s == "passed"),
        "rejected_delete_count": sum(1 for s in node_status.values() if s == "failed"),
        "not_processed_delete_count": sum(1 for s in node_status.values() if s == "not_processed"),
        "baseline_probe_status": baseline_probe_status if probes else "",
        "baseline_probe_error": str(baseline_probe.get("error", "")) if probes else "",
        "baseline_applied_probes": int(baseline_probe.get("applied_probes", 0) or 0) if probes else 0,
        "baseline_failed_to_apply": int(baseline_probe.get("failed_to_apply", 0) or 0) if probes else 0,
        "baseline_mutation_kills": baseline_kills if probes and baseline_probe_ok else "",
        "mutation_probe_count": len(probes),
        "post_suite_files": post_suite_files,
    }
    (out_dir / "strict_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "node_status": node_status,
        "node_note": node_note,
        "summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate test redundancy with empirical deselection, coverage, mutation, and branch-equivalence signals.",
    )
    ap.add_argument("--root", default=".", help="Repository root directory (default: current directory).")
    ap.add_argument("--python", default="python3", help="Python interpreter to use. Accepts a PATH name or an absolute path (default: python3).")
    ap.add_argument("--suite", action="append", default=[], help="Test file or directory to evaluate for redundancy (repeatable). At least one --suite is required.")
    ap.add_argument("--comparator-suite", action="append", default=[], help="Additional test file or directory used only for cross-suite overlap calculation (repeatable, optional).")
    ap.add_argument("--ranked-csv", default="", help="Path to a pre-computed ranked report CSV with coverage/mutation columns. When omitted, the script collects live coverage signal.")
    ap.add_argument("--inventory-csv", default="", help="Path to an inventory CSV with assertion-type overrides per test nodeid (optional).")
    ap.add_argument("--out-dir", default="artifacts/redundancy/api", help="Output directory for all generated artifacts (default: artifacts/redundancy/api).")
    ap.add_argument("--source-prefix", default="", help="Restrict coverage token collection to source files whose repo-relative path starts with this prefix (e.g. src/mypackage/). When omitted, coverage is collected from all source files.")
    ap.add_argument("--max-workers", type=int, default=max(1, min(os.cpu_count() or 2, 4)), help="Maximum parallel worker threads for candidate evaluation (default: min(cpu_count, 4)).")
    ap.add_argument("--timeout-seconds", type=int, default=900, help="Per-candidate pytest timeout in seconds (default: 900).")
    ap.add_argument("--strict-delete-gate", action="store_true", help="Enable the strict delete gate: repeated deselection, staged batch simulation, post-suite pass, and mutation-probe delta checks before confirming DELETE_SAFE_HIGH.")
    ap.add_argument("--strict-repeats", type=int, default=3, help="Number of repeated deselection runs required for strict gate stability (default: 3).")
    ap.add_argument("--strict-batch-size", type=int, default=8, help="Number of candidates to deselect together in each strict gate batch (default: 8).")
    ap.add_argument("--strict-max-batches", type=int, default=0, help="Maximum number of batches to process in the strict gate; 0 means unlimited (default: 0).")
    ap.add_argument("--strict-post-suite", action="append", default=[], help="Test file or directory to run as a full post-suite check after each strict gate batch (repeatable). Defaults to tests/ if it exists.")
    ap.add_argument("--strict-mutation-probes", type=int, default=3, help="Maximum number of mutation probes to use from --mutation-probes-config (default: 3).")
    ap.add_argument("--strict-mutation-max-drop", type=int, default=0, help="Maximum allowed drop in mutation kills between baseline and deselected batch (default: 0).")
    ap.add_argument(
        "--allow-numba-stub",
        action="store_true",
        help="Allow injecting a lightweight numba stub into PYTHONPATH when numba is missing. "
             "Disabled by default for repo-agnostic safety.",
    )
    ap.add_argument(
        "--mutation-probes-config",
        default=None,
        help="Path to a JSON file defining mutation probes for the strict gate. "
             "Format: [{\"probe_id\": \"P001\", \"file\": \"src/...\", \"old\": \"...\", \"new\": \"...\"}]. "
             "When omitted, no mutation probes are used and the mutation gate is skipped.",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment variable in KEY=VALUE format. Repeatable. "
             "Example: --env NUMBA_DISABLE_JIT=1 --env VIRTUAL_ENV=/path/to/venv",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.suite:
        raise SystemExit(
            "No suite files provided. Use --suite to specify test files or directories to analyse.\n"
            "Example: --suite tests/test_api.py --suite tests/test_core.py"
        )
    suite_files = resolve_and_validate_suite_paths(root, args.suite, arg_name="--suite")
    if not suite_files:
        raise SystemExit("None of the provided --suite paths exist under the repository root.")
    comparator_suite_files = resolve_and_validate_suite_paths(root, args.comparator_suite, arg_name="--comparator-suite")
    suite_set = set(suite_files)
    comparator_suite_files = [s for s in comparator_suite_files if s not in suite_set]
    comparator_suite_files = list(dict.fromkeys(comparator_suite_files))
    strict_post_suite_files = resolve_and_validate_suite_paths(root, args.strict_post_suite, arg_name="--strict-post-suite")
    strict_post_suite_files = list(dict.fromkeys(strict_post_suite_files))

    python_exe = resolve_python_exe(root, args.python)

    ranked_path = resolve_optional_path(root, args.ranked_csv)
    inventory_path = resolve_optional_path(root, args.inventory_csv)
    ranked_map = parse_ranked_by_nodeid(ranked_path) if ranked_path else {}

    mutation_probes_config_path = resolve_optional_path(root, args.mutation_probes_config)
    probes_source = (
        load_mutation_probes_from_config(mutation_probes_config_path)
        if mutation_probes_config_path
        else build_default_mutation_probes()
    )
    inv_assertions = parse_inventory_assertions(inventory_path) if inventory_path else {}

    tests = parse_test_metadata(root, suite_files)
    if not tests:
        raise SystemExit("No tests discovered from suite files.")

    # Enrich assertion surfaces from inventory if available.
    if inv_assertions:
        for t in tests:
            inv = inv_assertions.get(t.nodeid)
            if inv:
                t.assertion_types = inv

    # Parse --env KEY=VALUE pairs into extra environment variables.
    extra_env: dict[str, str] = {}
    for kv in args.env:
        key, _, value = kv.partition("=")
        if key:
            extra_env[key.strip()] = value.strip()

    env = build_runtime_env(
        root, out_dir, python_exe,
        allow_numba_stub=bool(args.allow_numba_stub),
        extra_env=extra_env or None,
    )
    use_xdist = has_xdist_plugin(root, python_exe, env)

    # Always materialize core evidence artifacts for downstream workflows.
    write_inventory_artifact(out_dir, tests)
    coverage_map, coverage_summary = write_coverage_artifacts(
        root,
        out_dir,
        tests,
        ranked_map,
        ranked_path,
        comparator_suite_files,
        python_exe=python_exe,
        env=env,
        timeout=args.timeout_seconds,
        max_workers=args.max_workers,
        source_prefix=args.source_prefix,
    )
    write_mutation_artifacts(out_dir, tests, ranked_map, ranked_path)

    by_cluster: dict[tuple[str, str], list[TestMeta]] = defaultdict(list)
    for t in tests:
        by_cluster[(t.entrypoint, t.intent)].append(t)

    # Exclude tests whose entrypoint could not be identified: all "unknown"
    # tests would share the same cluster key and could pair with completely
    # unrelated tests, producing false redundancy signals.
    candidates = [
        t for t in tests
        if t.entrypoint != "unknown" and len(by_cluster[(t.entrypoint, t.intent)]) > 1
    ]
    baseline = run_suite(
        root,
        python_exe,
        suite_files,
        env,
        deselect=None,
        timeout=args.timeout_seconds,
        use_xdist=use_xdist,
    )

    rows: list[dict[str, Any]] = []
    strict_gate_result: dict[str, Any] = {"node_status": {}, "node_note": {}, "summary": {}}
    if baseline["returncode"] != 0:
        rows.append(
            {
                "test_nodeid": "<baseline>",
                "validation_decision": "BASELINE_FAILED",
                "validation_reason": "Target suite failed before candidate evaluation",
                "deselect_suite_pass": False,
                "deselect_runtime_ms": round(baseline["runtime_ms"], 3),
                "deselect_failure_excerpt": baseline["output"][:1200],
                "strict_gate_status": "not_run",
                "strict_gate_note": "baseline failed",
            }
        )
    else:
        def peer_superset(t: TestMeta) -> bool:
            """True when a cluster peer's assertion types strictly dominate the candidate's.

            Also returns True for equal assertion type sets: if two tests assert
            the same things and one can be deselected, either is a deletion candidate.
            """
            peers = [p for p in by_cluster[(t.entrypoint, t.intent)] if p.nodeid != t.nodeid]
            for p in peers:
                if t.assertion_types and t.assertion_types.issubset(p.assertion_types):
                    return True
            return False

        def evaluate(t: TestMeta) -> dict[str, Any]:
            cluster = by_cluster[(t.entrypoint, t.intent)]
            cluster_size = len(cluster)
            run = run_suite(
                root,
                python_exe,
                suite_files,
                env,
                deselect=t.nodeid,
                timeout=args.timeout_seconds,
                use_xdist=use_xdist,
            )
            deselect_pass = run["returncode"] == 0
            ranked = ranked_map.get(t.nodeid, {})
            coverage_row = coverage_map.get(t.nodeid, {})
            low_signal = bool_low_signal(t, ranked, coverage_row)
            dominated = peer_superset(t)
            _rul = ranked.get("unique_line_count", "")
            report_unique_line = _rul if _rul != "" else coverage_row.get("unique_line_count", "")
            _rub = ranked.get("unique_branch_count", "")
            report_unique_branch = _rub if _rub != "" else coverage_row.get("unique_branch_count", "")
            _rov = ranked.get("cross_suite_overlap_ratio", "")
            report_overlap = _rov if _rov != "" else coverage_row.get("cross_suite_overlap_ratio", "")

            peers = [p for p in cluster if p.nodeid != t.nodeid]
            max_src_sim = max(
                (jaccard_sim(t.src_tokens, p.src_tokens) for p in peers), default=0.0
            )
            max_name_sim = max(
                (difflib.SequenceMatcher(None, t.test_name, p.test_name).ratio() for p in peers),
                default=0.0,
            )
            unique_mutability_contract = "mutability_contract" in t.assertion_types and not any(
                "mutability_contract" in p.assertion_types for p in peers
            )
            unique_exception_semantics = "exception" in t.assertion_types and not any(
                "exception" in p.assertion_types for p in peers
            )

            if t.is_parametrized:
                # Parametrized tests span multiple input variants; treat conservatively.
                decision = "KEEP_FOR_SIGNAL"
                reason = "parametrized test — evaluate variants individually before pruning"
            elif not deselect_pass:
                decision = "KEEP_FOR_STABILITY"
                reason = "suite failed when deselected"
            elif unique_mutability_contract:
                decision = "KEEP_FOR_SIGNAL"
                reason = "only test with mutability/readonly contract in this cluster"
            elif unique_exception_semantics:
                decision = "KEEP_FOR_SIGNAL"
                reason = "only test with explicit exception semantics in this cluster"
            elif low_signal and dominated:
                decision = "DELETE_SAFE_HIGH"
                reason = "deselect pass + dominated by peer + low unique signal"
            elif max_src_sim >= 0.90 and (low_signal or dominated):
                decision = "DELETE_SAFE_HIGH"
                reason = f"deselect pass + near-duplicate source (sim={max_src_sim:.2f}) + dominated/low-signal"
            elif dominated or max_src_sim >= 0.80 or max_name_sim >= 0.85:
                # Guard: if the nearest neighbour is already parametrized, merging
                # into it would create a false overlap signal after a prior merge.
                nearest_peer = max(
                    peers, key=lambda p: jaccard_sim(t.src_tokens, p.src_tokens)
                ) if peers else None
                if nearest_peer and nearest_peer.is_parametrized:
                    decision = "KEEP_FOR_SIGNAL"
                    reason = (
                        f"nearest neighbor ({nearest_peer.test_name}) is parametrized; "
                        "merge classification suppressed to avoid false overlap"
                    )
                else:
                    decision = "MERGE_RECOMMENDED"
                    reason = (
                        f"deselect pass + overlap candidate "
                        f"(dominated={dominated}, src_sim={max_src_sim:.2f}, name_sim={max_name_sim:.2f})"
                    )
            else:
                decision = "KEEP_FOR_SIGNAL"
                reason = "deselect pass + distinct source and assertion signal"

            return {
                "test_nodeid": t.nodeid,
                "entrypoint": t.entrypoint,
                "intent": t.intent,
                "cluster_size": cluster_size,
                "peer_superset_assertions": dominated,
                "assertion_types": ";".join(sorted(t.assertion_types)),
                "assert_count": t.assert_count,
                "is_parametrized": t.is_parametrized,
                "max_src_similarity": round(max_src_sim, 3),
                "max_name_similarity": round(max_name_sim, 3),
                "deselect_suite_pass": deselect_pass,
                "deselect_runtime_ms": round(run["runtime_ms"], 3),
                "report_unique_line_count": report_unique_line,
                "report_unique_branch_count": report_unique_branch,
                "report_mutants_unique_to_api": ranked.get("mutants_unique_to_api", ""),
                "report_overlap": report_overlap,
                "validation_decision": decision,
                "validation_reason": reason,
                "deselect_failure_excerpt": "" if deselect_pass else run["output"][:1200],
                "strict_gate_status": "pending",
                "strict_gate_note": "",
            }

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
            fut_to_nodeid = {ex.submit(evaluate, t): t.nodeid for t in candidates}
            for fut in as_completed(fut_to_nodeid):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    nodeid = fut_to_nodeid[fut]
                    rows.append({
                        "test_nodeid": nodeid,
                        "validation_decision": "KEEP_FOR_SIGNAL",
                        "validation_reason": f"evaluation error: {exc}",
                        "deselect_suite_pass": False,
                        "strict_gate_status": "not_run",
                        "strict_gate_note": "evaluation raised an exception",
                    })
        enforce_cluster_anchor(rows)

        if args.strict_delete_gate:
            strict_gate_result = run_strict_delete_gate(
                root,
                out_dir,
                python_exe,
                env,
                suite_files,
                strict_post_suite_files,
                args.timeout_seconds,
                use_xdist=use_xdist,
                rows=rows,
                repeats=max(1, args.strict_repeats),
                batch_size=max(1, args.strict_batch_size),
                max_batches=max(0, args.strict_max_batches),
                probes_source=probes_source,
                mutation_probe_count=max(0, args.strict_mutation_probes),
                mutation_max_drop=max(0, args.strict_mutation_max_drop),
            )
            node_status = strict_gate_result.get("node_status", {})
            node_note = strict_gate_result.get("node_note", {})
            for r in rows:
                nodeid = str(r.get("test_nodeid", ""))
                status = str(node_status.get(nodeid, "not_applicable"))
                note = str(node_note.get(nodeid, ""))
                r["strict_gate_status"] = status
                r["strict_gate_note"] = note
                if r.get("validation_decision") == "DELETE_SAFE_HIGH" and status != "passed":
                    r["validation_decision"] = "KEEP_FOR_SIGNAL"
                    if status == "not_processed":
                        r["validation_reason"] = "strict gate not completed for this candidate"
                    else:
                        r["validation_reason"] = f"strict gate blocked delete: {note or status}"
                elif r.get("validation_decision") == "DELETE_SAFE_HIGH" and status == "passed":
                    r["validation_reason"] = f"{r.get('validation_reason','')}; strict gate passed"
        else:
            for r in rows:
                r["strict_gate_status"] = "disabled"
                r["strict_gate_note"] = ""

    branch_map, branch_summary = write_branch_equiv_artifacts(
        root,
        out_dir,
        tests,
        by_cluster,
        rows,
        coverage_map,
        python_exe=python_exe,
        env=env,
        timeout=args.timeout_seconds,
        max_workers=args.max_workers,
        source_prefix=args.source_prefix,
    )
    for r in rows:
        nodeid = str(r.get("test_nodeid", ""))
        branch = branch_map.get(nodeid, {})
        r["branch_anchor_nodeid"] = branch.get("branch_anchor_nodeid", "")
        r["branch_exact_match"] = branch.get("branch_exact_match", "")
        r["branch_jaccard"] = branch.get("branch_jaccard", "")
        r["branch_candidate_only_count"] = branch.get("branch_candidate_only_count", "")
        r["branch_anchor_only_count"] = branch.get("branch_anchor_only_count", "")
        r["branch_status_note"] = branch.get("branch_status_note", "")

    # Re-evaluate DELETE_SAFE_HIGH candidates with branch-equivalence evidence.
    # A test that owns unique branches should not be flagged for deletion even
    # if assertion-type dominance passed during the initial evaluation pass.
    test_by_nodeid = {t.nodeid: t for t in tests}
    for r in rows:
        if r.get("validation_decision") != "DELETE_SAFE_HIGH":
            continue
        nodeid = str(r.get("test_nodeid", ""))
        branch = branch_map.get(nodeid)
        if not branch:
            continue
        t = test_by_nodeid.get(nodeid)
        if not t:
            continue
        ranked = ranked_map.get(nodeid, {})
        coverage_row = coverage_map.get(nodeid, {})
        if not bool_low_signal(t, ranked, coverage_row, branch_equiv_row=branch):
            r["validation_decision"] = "KEEP_FOR_SIGNAL"
            r["validation_reason"] = (
                f"branch-equiv re-evaluation: test has "
                f"{branch.get('branch_candidate_only_count', 0)} unique branches; "
                "not low-signal after branch-level analysis"
            )

    confidence_tier_counts = write_confidence_gate_artifact(out_dir, rows)

    rows.sort(key=lambda r: (r.get("validation_decision", ""), r.get("test_nodeid", "")))

    csv_headers = [
        "test_nodeid",
        "entrypoint",
        "intent",
        "cluster_size",
        "peer_superset_assertions",
        "assertion_types",
        "assert_count",
        "is_parametrized",
        "max_src_similarity",
        "max_name_similarity",
        "deselect_suite_pass",
        "deselect_runtime_ms",
        "report_unique_line_count",
        "report_unique_branch_count",
        "report_mutants_unique_to_api",
        "report_overlap",
        "validation_decision",
        "validation_reason",
        "deselect_failure_excerpt",
        "strict_gate_status",
        "strict_gate_note",
        "branch_anchor_nodeid",
        "branch_exact_match",
        "branch_jaccard",
        "branch_candidate_only_count",
        "branch_anchor_only_count",
        "branch_status_note",
        "gate_deselect",
        "gate_strict_stability",
        "gate_branch_exact",
        "gate_branch_high_similarity",
        "gate_assertion_dominance",
        "gate_unique_mutants",
        "gate_unique_coverage",
        "gate_cross_suite_overlap",
        "gate_external_side_effects",
        "gate_external_nonfunctional",
        "gate_external_history_regression",
        "confidence_tier",
        "manual_signoff_needed",
        "confidence_note",
    ]
    write_csv(out_dir / "candidate_validation.csv", rows, csv_headers)

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[str(r.get("validation_decision", ""))] += 1

    md = [
        "# Candidate Validation",
        "",
        f"Baseline suite pass: `{baseline['returncode'] == 0}`",
        "",
        "## Decision Counts",
        "",
    ]
    for k in sorted(counts):
        md.append(f"- `{k}`: {counts[k]}")

    md.extend(
        [
            "",
            "## Results",
            "",
            "| test_nodeid | entrypoint | intent | decision | confidence_tier | strict | branch_exact | branch_jaccard | deselect_pass | reason |",
            "|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for r in rows:
        md.append(
            "| "
            f"{r.get('test_nodeid','')} | {r.get('entrypoint','')} | {r.get('intent','')} | "
            f"{r.get('validation_decision','')} | {r.get('confidence_tier','')} | {r.get('strict_gate_status','')} | "
            f"{r.get('branch_exact_match','')} | {r.get('branch_jaccard','')} | {r.get('deselect_suite_pass','')} | "
            f"{r.get('validation_reason','')} |"
        )

    (out_dir / "candidate_validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "suite_files": suite_files,
        "comparator_suite_files": comparator_suite_files,
        "python_executable": python_exe,
        "xdist_enabled": use_xdist,
        "coverage_mode": coverage_summary.get("mode", ""),
        "strict_delete_gate": bool(args.strict_delete_gate),
        "strict_post_suite_files": strict_post_suite_files,
        "baseline_pass": baseline["returncode"] == 0,
        "candidates": len(candidates),
        "counts": dict(counts),
        "confidence_tier_counts": confidence_tier_counts,
        "strict_gate_summary": strict_gate_result.get("summary", {}),
        "branch_equiv_summary": branch_summary,
    }
    (out_dir / "candidate_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
