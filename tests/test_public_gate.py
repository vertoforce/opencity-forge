"""The public-release gate (tools/check_public.py) passes on the whole tree, and its rules
catch the shapes they exist for.  Samples are synthetic; the address samples are split
into two literals so this file's own lines carry no address."""
import os, struct, subprocess, sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, 'tools', 'check_public.py')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import check_public as G  # noqa: E402


def test_tree_passes_the_public_gate():
    r = subprocess.run([sys.executable, GATE, '--root', ROOT],
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]


def test_local_denylist_is_gitignored():
    r = subprocess.run(['git', 'check-ignore', '-q', '.private-denylist'], cwd=ROOT)
    assert r.returncode == 0


# one sample line per shape; each must give at least one hit of the named rule
SAMPLES = [
    ('personal:email', 'contact a@' 'b.example for access.'),
    ('personal:ip', 'the box answers on 10.0.' '0.7 only.'),
    ('personal:mac', 'the card is 00:11:22' ':33:44:55 here.'),
    ('personal:host-path', 'files sit under /ho' 'me/someone/grid.'),
    ('personal:hostname', 'reach it at box.lo' 'cal for now.'),
    ('personal:path', 'the tiles are in /opt/tiles/grid.'),
    ('personal:date', 'built on 2020-01-02 with the grid.'),
    ('personal:date', 'the cap holds since 01-02 for the grid.'),
    ('personal:date', 'the grid was set yesterday.'),
    ('personal:year', 'the 2021 grid is coarse.'),
    ('personal:clock', 'the job ran at 14:05 on the grid.'),
    ('personal:first-person', 'we decided to keep the grid.'),
    ('personal:first-person', 'the grid was set in this session.'),
    ('personal:first-person', 'the owner wants a finer grid.'),
    ('personal:first-person', 'per the user, the grid is fixed.'),
    ('personal:first-person', 'the grid gives us a frame.'),
    ('personal:first-person', 'I measured the grid.'),
    ('personal:quote', 'the tester said: "too dark".'),
    ('personal:history', 'Replaced by GRID_RULE.'),
    ('personal:history', 'the grid used to be finer.'),
    ('personal:history', 'the cap no longer applies.'),
    ('personal:history', 'the flag is kept for backward runs.'),
    ('personal:history', 'this is dead code.'),
    ('personal:history', 'the sum is one, as it always was.'),
    ('personal:history', 'the grid rule is left as it was.'),
    ('personal:history', 'the cap is unchanged from stage 3.'),
    ('personal:history', 'after the fix the grid is finer.'),
    ('personal:history', 'the cap was set in round 3.'),
    ('personal:hardware', 'the grid needs 16 GB of VRAM.'),
    ('personal:history', 'measured over the views after the tone, the grid is finer.'),
    ('personal:history', 'the exponent comes from before/after pairs on the grid.'),
    ('personal:history', 'the sd on the grid drops from 2.11 to 0.25.'),
    ('personal:private-tool', 'no stage in this repository writes the grid.'),
    ('personal:private-tool', 'an external tool can supply the grid.'),
    ('fragment:empty parens', 'the value ( ) is kept.'),
    ('fragment:dangling see', 'the grid is coarse (see ).'),
    ('fragment:space before full stop', 'the grid is kept .'),
    ('fragment:double comma', 'the grid, , the cap.'),
    ('fragment:empty aside', 'the cap (since) holds.'),
    ('fragment:cut subject', 'the grid stays when decided by size.'),
    ('fragment:dangling function word', 'the grid is finer than.'),
    ('fragment:cut punctuation', 'the grid,. is kept.'),
    ('fragment:doubled word', 'the the grid is kept.'),
    ('fragment:article swap', 'a apple sits on the grid.'),
    ('fragment:dangling last word', 'the grid is kept by the'),
    ('cited:missing file', 'the grid comes from lib/nothing_here.py.'),
    ('cited:missing stage', 'stage 77 writes the grid.'),
    ('cited:missing module', 'the grid comes from s77_grid.'),
]
CLEAN = ['the pole stands 30 m from the wall.', 'the US border.', 'US Geological Survey lidar.',
         'a run no longer than four texels is filled.', 'the colour is replaced by the class mean.',
         'the grid is used to place trees.', 'the swing is unchanged and only the scale moves.']


def _scan_text(tmp_path, text, name='doc.md'):
    (tmp_path / name).write_text(text + '\n')
    return G.scan(str(tmp_path), quiet=True)


@pytest.mark.parametrize('rule,line', SAMPLES)
def test_rule_catches_its_shape(tmp_path, rule, line):
    hits = _scan_text(tmp_path, line)
    assert any(h[2] == rule for h in hits), (line, hits)


@pytest.mark.parametrize('line', CLEAN)
def test_rule_leaves_plain_prose_alone(tmp_path, line):
    assert _scan_text(tmp_path, line) == []


def test_python_code_gets_address_rules_and_comments_get_prose_rules(tmp_path):
    hits = _scan_text(tmp_path, "HOST = 'box.lo" "cal'\n"
                                "N = 2020  # the grid; we decided\n# , then the cap\nX = 1\n", 'm.py')
    rules = {h[2] for h in hits}
    assert {'personal:hostname', 'personal:first-person', 'fragment:leading punctuation'} <= rules, hits
    assert 'personal:year' not in rules, hits                    # code, not prose


def test_code_names_after_a_revision_or_a_fix_are_caught(tmp_path):
    # the names are glued at run time so this file does not carry them itself
    text = ("fp = 'grid_colour.jpg'\nfp = 'grid_colour_v" + "2.jpg'\n"
            "OPT = '--no-grid-tint'\nOPT = '--no-grid" + "fix'\nP = 'prefix'\n")
    hits = _scan_text(tmp_path, text, 'm.py')
    lines = sorted(h[1] for h in hits if h[2] == 'personal:history')
    assert lines == [2, 4], hits


def test_dates_in_code_numbers_and_seeds_are_caught(tmp_path):
    # the dates are glued at run time so this file does not carry them itself
    text = ("rng = np.random.default_rng(" + "20" + "200102)\n"
            "r = np.random.default_rng((oid * 7 + " + "20" + "20) % 97)\n"
            "rng = np.random.default_rng(81)\nN = 2020\n")
    hits = _scan_text(tmp_path, text, 'm.py')
    lines = sorted(h[1] for h in hits if h[2] == 'personal:date')
    assert lines == [1, 2], hits


def test_denylist_lines_and_split_names(tmp_path, monkeypatch):
    monkeypatch.delenv('OCF_PRIVATE_DENYLIST', raising=False)
    deny = tmp_path / 'deny.txt'
    deny.write_text('# local names\nzorblax\nquux\\s+corge\n')
    root = tmp_path / 'r'
    root.mkdir()
    (root / 'doc.md').write_text('the zorblax grid.\nit ends in quux\ncorge starts here.\n')
    hits = G.scan(str(root), quiet=True, deny_path=str(deny))
    assert sorted(h[1] for h in hits if h[2] == 'denylist') == [1, 2], hits
    assert G.scan(str(root), quiet=True, require_denylist=True)[0][2] == 'denylist:missing'


def test_empty_section_header_is_caught(tmp_path):
    hits = _scan_text(tmp_path, 'x = 1\n\n\n# ---------- the atlas ----- #\n'
                                '# ---------- the grid ----- #\ndef f():\n    return 1\n\n\n'
                                '# ---------- stages ----- #', 'm.py')
    assert sorted(h[1] for h in hits if h[2] == 'fragment:empty section') == [4, 10], hits


def test_image_metadata_is_caught(tmp_path):
    chunk = struct.pack('>I', 4) + b'tEXt' + b'a\0bc' + b'\0\0\0\0'
    (tmp_path / 'x.png').write_bytes(b'\x89PNG\r\n\x1a\n' + chunk)
    assert any(h[2] == 'binary:image metadata' for h in G.scan(str(tmp_path), quiet=True))


def test_commits_and_trees_are_scanned(tmp_path):
    repo = tmp_path / 'r'
    repo.mkdir()
    ident = ['-c', 'user.name=gate-test', '-c', 'user.email=noreply' + '@example.invalid']
    env = dict(os.environ, GIT_AUTHOR_DATE='2000-01-01T00:00:00+02:00',
               GIT_COMMITTER_DATE='2000-01-01T00:00:00+00:00')
    (repo / 'leak.md').write_text('our grid is coarse.\n')
    for cmd in (['init', '-q'], ['add', 'leak.md'], ['commit', '-q', '-m', 'we decided to add it']):
        subprocess.run(['git', *ident, *cmd], cwd=repo, check=True, env=env, capture_output=True)
    (repo / 'leak.md').unlink()
    assert G.scan(str(repo), quiet=True) == []
    trees = G.scan_trees(str(repo), ['HEAD'], quiet=True)
    assert any(h[0].endswith(':leak.md') and h[2] == 'personal:first-person' for h in trees), trees
    rules = {h[2] for h in G.scan(str(repo), quiet=True, commits=['HEAD'])}
    assert {'commit:non-UTC date', 'personal:first-person'} <= rules, rules
