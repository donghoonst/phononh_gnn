"""Verify source parity and execute the notebook with project imports blocked."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/IFC2_All_in_One_Korean.ipynb'


class NotebookSourceTransform(ast.NodeTransformer):
    def __init__(self, rename=None):
        self.rename = rename

    def visit_ImportFrom(self, node):
        if node.level or (node.module or '').startswith('ifc2'):
            return None
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == 'main' and self.rename:
            node.name = self.rename
            node.args.args.append(ast.arg(arg='argv'))
            node.args.defaults.append(ast.Constant(None))
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if self.rename and isinstance(node.func, ast.Attribute) and node.func.attr == 'parse_args' and not node.args:
            node.args.append(ast.Name(id='argv', ctx=ast.Load()))
        return node


def validate_source(nb):
    for relative, expected in nb.metadata.ifc2_notebook.source_sha256.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    exports = set()
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        compile(tree, '<notebook-cell>', 'exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.level and not (node.module or '').startswith('ifc2')
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith('ifc2') for alias in node.names)
        relative = cell.metadata.get('source_file')
        if not relative:
            continue
        symbol = cell.metadata.symbol
        source_name = 'main' if symbol in {'ifc2_cli_main', 'fetch_zenodo_cli'} else symbol
        original = ast.parse((ROOT / relative).read_text())
        if source_name == 'SLICES':
            node = next(n for n in original.body if isinstance(n, ast.Assign))
        else:
            node = next(n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == source_name)
        normalized = NotebookSourceTransform(symbol if source_name == 'main' else None).visit(node)
        assert ast.dump(normalized, include_attributes=False) == ast.dump(tree.body[0], include_attributes=False), (relative, symbol)
        exports.add((relative, source_name))
    for relative in nb.metadata.ifc2_notebook.source_sha256:
        for node in ast.parse((ROOT / relative).read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if relative.startswith('tests/') and node.name in {'sample', 'numeric_defaults'}:
                    continue
                assert (relative, node.name) in exports, (relative, node.name)
    return len(exports)


def main():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    count = validate_source(nb)
    print(f'Source parity verified: {count} definitions/constants', flush=True)
    guard = nbformat.v4.new_code_cell('''
import sys
import importlib.abc
class BlockProjectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ifc2" or fullname.startswith("ifc2."):
            raise ImportError("Validation forbids project imports: " + fullname)
sys.meta_path.insert(0, BlockProjectImports())
''')
    nb.cells.insert(0, guard)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='ifc2_notebook_standalone_') as tmp:
        work = Path(tmp)
        env = os.environ.copy()
        for name, subdir in [('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('JUPYTER_CONFIG_DIR','jupyter_config'),
                             ('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','cache')]:
            path = work / subdir
            path.mkdir()
            env[name] = str(path)
        env['OMP_NUM_THREADS'] = '1'
        env['MPLBACKEND'] = 'Agg'
        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == 'code' and cell.metadata.get('tags'):
                print(f'Running cell {cell_index}: {cell.metadata.tags}', flush=True)
        client = NotebookClient(nb, timeout=900, kernel_name='python3',
                                resources={'metadata':{'path':str(work)}}, on_cell_start=progress)
        client.execute(cwd=str(work), env=env)
        summaries = list(work.glob('ifc2_notebook_runs/*/notebook_summary.json'))
        assert len(summaries) == 1
        summary = json.loads(summaries[0].read_text())
        assert summary['tests_passed'] == 17
        assert len(summary['ablation_models_run']) == 8
        assert summary['local_ifc2_package_imported'] is False
        assert not summary['real_data_training_run'] and not summary['zenodo_listing_run']
        output_dir = ROOT / 'artifacts/notebook_validation'
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(summaries[0].parent, output_dir / 'run', dirs_exist_ok=True)
    nb.cells.pop(0)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None
            assert all(output.output_type != 'error' for output in cell.outputs)
    nbformat.write(nb, NOTEBOOK)
    report = dict(notebook=str(NOTEBOOK.relative_to(ROOT)), code_cells=sum(c.cell_type=='code' for c in nb.cells),
                  total_cells=len(nb.cells), source_definitions_verified=count,
                  project_imports_blocked=True, independent_working_directory=True,
                  elapsed_seconds=time.perf_counter()-started, **summary)
    (ROOT/'docs/notebook_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
