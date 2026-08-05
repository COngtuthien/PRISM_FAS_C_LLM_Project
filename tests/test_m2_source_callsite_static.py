import ast
from pathlib import Path


def test_run_preprocessing_source_routing_callsite_contract():
    runner_path = Path(__file__).parents[1] / "src" / "prism_fas" / "data" / "m2_runner.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    run_preprocessing = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_preprocessing"
    )

    calls = [
        node
        for node in ast.walk(run_preprocessing)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "route_source_success"
    ]
    assert len(calls) == 1
    routing_call = calls[0]
    keyword_names = {keyword.arg for keyword in routing_call.keywords if keyword.arg is not None}
    assert {
        "repository",
        "context",
        "canonical_record",
        "sample_id",
        "requested_frame_index",
        "actual_frame_index",
        "bbox",
        "landmarks",
        "crop_relative_path",
        "crop_sha256",
    } <= keyword_names

    source_guards = [
        node
        for node in ast.walk(run_preprocessing)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.left, ast.Attribute)
        and isinstance(node.test.left.value, ast.Name)
        and node.test.left.value.id == "context"
        and node.test.left.attr == "dataset_role"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "source"
    ]
    assert len(source_guards) == 1
    source_guard = source_guards[0]
    assert routing_call in ast.walk(source_guard)

    runtime_errors = [
        node
        for node in ast.walk(source_guard)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeError"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "source routing requires an initialized manifest repository"
    ]
    assert runtime_errors

    counter_increments = [
        node
        for node in ast.walk(run_preprocessing)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"success", "crops"}
        and isinstance(node.op, ast.Add)
        and node.lineno > routing_call.lineno
    ]
    assert {increment.target.id for increment in counter_increments} == {"success", "crops"}

    direct_converter_calls = [
        node
        for node in ast.walk(run_preprocessing)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"build_source_frame_record", "build_source_crop_record"}
    ]
    assert not direct_converter_calls
