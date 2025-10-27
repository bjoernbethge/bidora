# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "bpy",
# ]
# ///
"""AST-based dataset generator for Blender/3D Python code.

Extracts structured training data from Python codebases using AST parsing.
Generates instruction-response pairs for fine-tuning LLMs on 3D code generation.
"""

import ast
import json
from pathlib import Path
from typing import Iterator


def extract_function_docstrings(code: str) -> Iterator[dict[str, str]]:
    """Extract functions with docstrings as instruction-output pairs.

    Args:
        code: Python source code

    Yields:
        Dict with 'instruction' (from docstring) and 'output' (function code)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                # Generate instruction from docstring
                instruction = f"Write a Python function that {docstring.split('.')[0].lower()}"

                # Get function code
                function_code = ast.unparse(node)

                yield {
                    "instruction": instruction,
                    "output": function_code
                }


def extract_all_functions(code: str) -> Iterator[dict[str, str]]:
    """Extract ALL functions (with or without docstrings).

    Args:
        code: Python source code

    Yields:
        Dict with 'instruction' and 'output' for all functions
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get function signature
            func_name = node.name
            args = ast.unparse(node.args) if hasattr(node, 'args') else ''

            # Create instruction based on function name
            instruction = f"Write a Python function named {func_name}"

            # Get function code
            function_code = ast.unparse(node)

            yield {
                "instruction": instruction,
                "output": function_code
            }


def extract_blender_operations(code: str) -> Iterator[dict[str, str]]:
    """Extract ALL Blender API calls (bpy.*) as instruction-output pairs.

    Args:
        code: Python source code with Blender API calls

    Yields:
        Dict with 'instruction' and 'output' for Blender operations
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        # Find ALL bpy.* calls (not just ops)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                func_name = ast.unparse(node.func)

                if func_name.startswith("bpy."):
                    # Extract operation name
                    parts = func_name.replace("bpy.", "").split(".")
                    op_name = " ".join(parts).replace("_", " ")
                    instruction = f"Create Blender code using {func_name}"

                    # Get full statement
                    output = ast.unparse(node)

                    yield {
                        "instruction": instruction,
                        "output": output
                    }


def extract_class_definitions(code: str) -> Iterator[dict[str, str]]:
    """Extract class definitions as instruction-output pairs.

    Args:
        code: Python source code

    Yields:
        Dict with 'instruction' (class purpose) and 'output' (class code)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node)
            class_name = node.name

            # Generate instruction
            if docstring:
                instruction = f"Define a Python class {class_name} that {docstring.split('.')[0].lower()}"
            else:
                instruction = f"Define a Python class named {class_name}"

            # Get class code
            class_code = ast.unparse(node)

            yield {
                "instruction": instruction,
                "output": class_code
            }


def extract_bpy_property_access(code: str) -> Iterator[dict[str, str]]:
    """Extract bpy.* property access patterns.

    Args:
        code: Python source code

    Yields:
        Dict with property access examples
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        # Find attribute access like obj.location, mesh.vertices, etc
        if isinstance(node, ast.Attribute):
            attr_code = ast.unparse(node)
            if 'bpy' in attr_code or any(kw in attr_code for kw in ['data', 'context', 'types', 'props']):
                instruction = f"Access Blender property {attr_code}"
                yield {
                    "instruction": instruction,
                    "output": attr_code
                }


def extract_imports(code: str) -> Iterator[dict[str, str]]:
    """Extract import statements.

    Args:
        code: Python source code

    Yields:
        Dict with import examples
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_code = ast.unparse(node)
            if 'bpy' in import_code:
                instruction = f"Import Blender module"
                yield {
                    "instruction": instruction,
                    "output": import_code
                }


def code_completion_pairs(code: str) -> Iterator[dict[str, str]]:
    """Generate code completion pairs by removing random AST subtrees.

    Args:
        code: Python source code

    Yields:
        Dict with 'prompt' (incomplete code) and 'completion' (missing part)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    # Find function bodies
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.body:
            # Split function: signature + first half of body vs rest
            mid = len(node.body) // 2
            if mid > 0:
                # Create incomplete function
                incomplete_node = ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=node.body[:mid] + [ast.Pass()],
                    decorator_list=node.decorator_list,
                    returns=node.returns,
                )
                # Copy location information from original node
                ast.copy_location(incomplete_node, node)

                prompt = ast.unparse(incomplete_node)
                completion = "\n".join([ast.unparse(stmt) for stmt in node.body[mid:]])

                yield {
                    "prompt": prompt,
                    "completion": completion
                }


def scan_directory(directory: Path, pattern: str = "*.py") -> Iterator[dict[str, str]]:
    """Scan directory for Python files and extract training data.

    Args:
        directory: Directory to scan
        pattern: File glob pattern

    Yields:
        Training examples from all matching files
    """
    for file_path in directory.rglob(pattern):
        try:
            code = file_path.read_text(encoding="utf-8")

            # Extract different types of patterns
            yield from extract_function_docstrings(code)  # Functions WITH docstrings
            yield from extract_all_functions(code)  # ALL functions
            yield from extract_blender_operations(code)  # ALL bpy.* calls
            yield from extract_class_definitions(code)  # Class definitions
            yield from extract_bpy_property_access(code)  # Property access
            yield from extract_imports(code)  # Import statements
            yield from code_completion_pairs(code)  # Code completion

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def generate_dataset(
    source_dirs: list[Path],
    output_file: Path,
    min_examples: int = 1000
) -> None:
    """Generate JSONL dataset from Python source directories.

    Args:
        source_dirs: List of directories containing Python source code
        output_file: Output JSONL file path
        min_examples: Minimum number of examples to generate
    """
    examples = []
    source_stats = {}  # Track examples per source

    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: {source_dir} does not exist, skipping")
            continue

        print(f"Scanning {source_dir}...")
        source_name = source_dir.name
        start_count = len(examples)

        for example in scan_directory(source_dir):
            examples.append(example)

            # Only show major milestones
            if len(examples) in [100, 500, 1000, 2000, 5000]:
                print(f"  Extracted {len(examples)} examples...")

        # Track per-source stats
        source_stats[source_name] = len(examples) - start_count

    # Deduplicate by output
    unique_examples = []
    seen_outputs = set()

    for ex in examples:
        output_key = ex.get("output", ex.get("completion", ""))
        if output_key not in seen_outputs:
            unique_examples.append(ex)
            seen_outputs.add(output_key)

    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print('='*60)
    print(f"Total examples extracted: {len(examples)}")
    print(f"Unique examples: {len(unique_examples)}")
    print(f"Deduplication rate: {100 * (1 - len(unique_examples)/len(examples)):.1f}%")

    print(f"\nPer-source breakdown:")
    for source_name, count in source_stats.items():
        print(f"  {source_name}: {count} examples")

    if len(unique_examples) < min_examples:
        print(f"\nWarning: Only {len(unique_examples)} examples generated (target: {min_examples})")
    else:
        print(f"\nTarget reached! ({len(unique_examples)} >= {min_examples})")

    # Write JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for example in unique_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nDataset saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")


def main() -> None:
    """Generate dataset from bpy package and Blender extensions."""

    print("AST-based Dataset Generator for Blender/3D Code")
    print("=" * 60)

    source_dirs = []

    # 1. Find bpy package
    try:
        import bpy
        bpy_path = Path(bpy.__file__).parent
        source_dirs.append(bpy_path)
        print(f"✓ Found bpy package: {bpy_path}")
    except ImportError:
        print("✗ bpy package not found (install with: pip install bpy)")

    # 2. Find Blender extensions
    extensions_path = Path.home() / "AppData/Roaming/Blender Foundation/Blender/4.5/extensions/blender_org"
    if extensions_path.exists():
        source_dirs.append(extensions_path)
        print(f"✓ Found Blender extensions: {extensions_path}")
    else:
        print(f"✗ Blender extensions not found at: {extensions_path}")

    # 3. Optional: Blender core installation
    blender_core = Path("C:/Program Files/Blender Foundation/Blender/4.5/4.5/scripts/addons")
    if blender_core.exists():
        source_dirs.append(blender_core)
        print(f"✓ Found Blender core addons: {blender_core}")

    if not source_dirs:
        print("\nError: No source directories found!")
        print("Please install bpy or Blender 4.5")
        return

    print(f"\nScanning {len(source_dirs)} source director{'y' if len(source_dirs) == 1 else 'ies'}...")
    print()

    output_file = Path("data/blender_dataset.jsonl")
    generate_dataset(
        source_dirs=source_dirs,
        output_file=output_file,
        min_examples=1000
    )

    print("\nDataset generation complete!")
    print(f"\nTo train with this dataset:")
    print(f"  bidora train --train-file {output_file} --val-file {output_file} \\")
    print(f"    --model Qwen/Qwen3-4B --rank 8 --epochs 3")


if __name__ == "__main__":
    main()
