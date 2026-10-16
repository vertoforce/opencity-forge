"""The README and docs/HOW_IT_WORKS.md describe the commands and the stage order that
pipeline/area_cli.py actually has."""
import os, re

import area_cli as CLI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


def test_readme_lists_every_command():
    text = _read('README.md')
    missing = [c for c in CLI.COMMANDS if f'| `{c}` |' not in text]
    assert not missing, missing


def _chain_index():
    idx = {}
    for i, (script, _) in enumerate(CLI.POSES_STAGES):
        idx.setdefault(os.path.splitext(os.path.basename(script))[0], i)
    return idx


def test_stage_table_follows_the_run_order():
    text = _read('docs/HOW_IT_WORKS.md')
    body = text.split('## The stages, in order', 1)[1].split('\n## ', 1)[0]
    idx = _chain_index()
    order = []
    for row in (l for l in body.split('\n') if l.startswith('| ') and not l.startswith('| Step')):
        cells = row.split('|')
        names = re.findall(r'\b(s\d{2}[a-z]?_\w+|nomesh_\w+|city_props|objassets)\b', cells[2])
        hit = [idx[n] for n in names if n in idx]
        if hit:
            order.append((min(hit), cells[1].strip()))
    assert len(order) >= 6, order
    assert [i for i, _ in order] == sorted(i for i, _ in order), order


def test_knobs_are_derived_before_the_stage_chain():
    assert CLI.POSES_STEPS.index('derive') < CLI.POSES_STEPS.index('stages')
    text = _read('docs/HOW_IT_WORKS.md')
    rows = [l.split('|')[1].strip() for l in text.split('## The stages, in order', 1)[1].split('\n')
            if l.startswith('| ') and not l.startswith('| Step')]
    assert rows.index('Knobs') < rows.index('Context'), rows
