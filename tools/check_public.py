#!/usr/bin/env python3
"""Public-release gate: fail when a file in the tree carries personal or private content.

Scans every tracked (and untracked, not ignored) file and prints each hit as
`path:line: rule: text`.  Exit 1 on any hit.  Stdlib only.

Rule families:

  personal  email addresses, private IPv4 addresses, MAC addresses, absolute host paths and
            LAN host names (every line); dates, clock times, first-person or credited
            decisions, attributed quotes, development-history wording and hardware model
            names (prose only)
  fragment  shapes an edit leaves when it cuts words out of a sentence, and a section
            header with no code under it
  denylist  every regex in a local, untracked file (OCF_PRIVATE_DENYLIST, else
            ./.private-denylist); skipped with a notice when the file is absent
  cited     a repo file, stage or module named in prose that is not in the tree
  binary    a .glb's JSON chunk (address shapes and denylist); EXIF / XMP / text chunks in images
  commit    with --commits: author, committer, non-UTC dates and message of each commit
  tree      with --trees: every rule on the committed file tree of each commit reachable
            from the given revisions

Prose is a .py file's comments and string literals, and every line of other text files.

    python3 tools/check_public.py [--root DIR] [--quiet] [--require-denylist]
                                  [--commits [REV ...]] [--trees [REV ...]]
"""
import argparse, fnmatch, io, os, re, struct, subprocess, sys, tokenize

BINARY_EXT = {'.glb', '.gltf', '.jpg', '.jpeg', '.png', '.gif', '.webp', '.tif', '.tiff', '.npz', '.npy',
              '.ktx2', '.bin', '.zip', '.gz', '.pdf'}

# ---- personal: every line, code included ------------------------------------------------
EMAIL = re.compile(r'[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
PRIVATE_IP = re.compile(r'(?<![\w.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])'
                        r'(?:\.\d{1,3}){2}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2})(?![\w.]*\d)')
MAC = re.compile(r'\b[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}\b')
HOST_ROOT = re.compile(r'(?<![\w.$}~:/-])/(?:root|home|mnt|scratch|srv|media|Users|nfs|export|volume\d*|data\d+)/')
HOSTNAME = re.compile(r'\b(?!self\.|threading\.|cls\.)[\w-]+\.(?:local|lan|home|internal|localdomain|intranet)\b(?![\w(])')
LINE_RULES = (('personal:email', EMAIL), ('personal:ip', PRIVATE_IP), ('personal:mac', MAC),
              ('personal:host-path', HOST_ROOT), ('personal:hostname', HOSTNAME))
# ---- history: every line of a tracked file, code included -----------------------------
# names that carry a revision or a fix instead of saying what the thing is
VERSION_NAME = re.compile(r'\b[A-Za-z]\w*_v\d+\.(?:jpe?g|png|webp|npz|npy|json|glb|ktx2)\b')
FIX_NAME = re.compile(r'\b(?!(?:pre|suf|af|in)fix)[a-z]{3,}[_-]?fix(?:es|ed)?\b|\b[a-z]+[_-]?fix[_-][a-z]+')
# a calendar date written as a number (YYYYMMDD), and a year used as a random seed or salt
DATE_NUMBER = re.compile(r'(?<![\w.])(?:19|20)\d\d(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?![\w.])')
SEED_YEAR = re.compile(r'\b(?:default_rng|RandomState|SeedSequence|seed)\b[^#\n]*?(?<![\w.])(?:19|20)\d\d(?![\w.])')
CODE_RULES = (('personal:history', VERSION_NAME), ('personal:history', FIX_NAME),
              ('personal:date', DATE_NUMBER), ('personal:date', SEED_YEAR))

# ---- personal: prose ---------------------------------------------------------------------
# an absolute path outside the system dirs every reader also has
ABS_PATH = re.compile(r'(?<![\w.$}~:/<>*\'"`=+@-])/(?!(?:usr|tmp|dev|proc|bin|etc|sys)/)[A-Za-z_][\w.-]*/[\w.-]')
DATE = re.compile(r'\b(?:19|20)\d\d-[01]\d(?:-[0-3]\d)?\b|\b(?:19|20)\d\d/[01]\d/[0-3]\d\b'
                  r'|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b'
                  r'|\b(?:since|on|from|until|before|after|by)\s+[01]?\d[-/][0-3]\d\b')
YEAR = re.compile(r'(?<![\w.-])20[12]\d(?![\w.-]*\d)')
CLOCK = re.compile(r'(?<![\w:.\[])(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*(?:am|pm|AM|PM|UTC))?(?![\w:\]])')
TIMELINE = re.compile(r'\b(?:tod[a]y|yesterd[a]y|tonig[h]t|last\s+(?:night|week|month)|this\s+(?:morning|week|month))\b', re.I)
FIRST_PERSON = re.compile(
    r'\b(?:this|last|that)\s+session\b|\bwe\s+(?:decided|chose|agreed|picked|measured|found|tried)\b'
    r'|\bper\s+(?:the\s+)?user\b|\b(?:the\s+)?user\s+(?:said|asked|wants|wanted|decided|prefers|requested)\b'
    r'|\bthe\s+(?:owner|designer)\b|\b(?:owner|designer)\'s\b|\bmy\s+(?:setup|machine|box|server|lab|home|network)\b'
    r'|\bhome\s*lab\b', re.I)
# first-person pronouns; case-sensitive, so the country code US stays out
FIRST_PERSON_WE = re.compile(r'\b(?:[Ww]e|us|[Oo]ur|[Oo]urs|[Oo]urselves)\b(?![\w(=.\'-])')
FIRST_PERSON_I = re.compile(r'\bI\s+(?:measured|think|found|tried|fixed|saw|ran|decided|chose|built|wrote|noticed)\b')
QUOTE = re.compile(r'\b(?:said|says|asked|wrote|complained|commented|reported)\s*[:,]?\s+["\u201c]\w'
                   r'|\b(?:reviewers?|testers?)\s+(?:said|wrote|asked|complained|called)\b'
                   r'|(?-i:[\w.!?]["\u201d]\s*[\u2014\u2013-]{1,2}\s*(?:the\s+)?[A-Z][a-z]+\b)', re.I)
# wording that narrates how the code got here instead of saying what it does
HISTORY = re.compile(
    r'(?<!\bis\s)(?<!\bare\s)(?<!\bbe\s)\breplaced\s+by\b|\bsuperseded\s+by\b|\bwas\s+renamed\b|\bthis\s+replaces\b'
    r'|(?<!\bis\s)(?<!\bare\s)(?<!\bbe\s)(?<!\bwas\s)(?<!\bget\s)\bused\s+to\b'
    r'|\bno\s+longer\b(?!\s+than)|\bany\s?more\b(?!\s+(?:than|of|to))|\b(?:exactly\s+)?as\s+before\b'
    r'|\bas\s+(?:it|they)\s+always\s+(?:was|were|did|has|have|had)\b|\balways\s+was\b'
    r'|\bwhat\s+(?:it|they)\s+(?:was|were)\b|\bdead\s+(?:code|branch)\b'
    r'|#\s*unchanged\b|\bunchanged\s+from\b|\b(?:left|kept)\s+unchanged\b'
    r'|\b(?:left|stays?|kept)\s+as\s+(?:it|they)\s+(?:was|were)\b'
    r'|\bkept\s+(?:for\s+(?:the\s+)?(?:old|backward|history|record)|as\s+the\s+fallback)\b'
    r'|\b(?:pre|post)-fix\b|\b(?:after|before)\s+the\s+fix\b|(?-i:\bFIX\b\s*[:,.]|\bFix\s*[:,]|^\s*fix:)'
    r'|\bthe\s+old\s+(?:rule|value|default|version|code|layout|path)\b'
    r'|\b(?:earlier|previous|first|this|next)\s+rounds?\b|\bround\s+\d'
    r'|\bafter\s+the\s+(?:tone|tuning|retune|change|patch|rework)\b|\bbefore/after\s+pairs\b'
    r'|\b(?:drops?|dropped|fell|falls?|rises?|rose|moves?|moved|went|goes)\s+(?:[\w-]+\s+){0,3}from\s+\(?-?\d[\d.]*\)?'
    r'[^.;]{0,25}?\s+to\s+\(?-?\d', re.I)
# pointers to tooling the reader does not have
PRIVATE_TOOL = re.compile(r'\b(?:no|none\s+is|not)\s+(?:[\w-]+\s+){0,3}(?:in|by|from)\s+this\s+repo(?:sitory)?\b'
                          r'|\ban\s+external\s+tool\b', re.I)
HARDWARE = re.compile(
    r'\b(?:R[T]X|G[T]X|Quad[r]o|Rade[o]n|Ryz[e]n|Xe[o]n|EP[Y]C|Threadrip[p]er|GeF[o]rce|A1[0]0|H1[0]0)\b'
    r'|\bCore\s+i[3579]\b|\bApple\s+M[1-4]\b'
    r'|\b\d+(?:\.\d+)?\s*(?:GB|GiB|TB|TiB)\s+(?:of\s+)?(?:RAM|VRAM|memory|disk|SSD|NVMe|HDD)\b', re.I)
LICENCE_LINE = re.compile(r'copyright|\(c\)|\u00a9|licen[cs]e', re.I)     # a year is fine here
PROSE_RULES = (('personal:date', DATE), ('personal:year', YEAR), ('personal:clock', CLOCK),
               ('personal:date', TIMELINE), ('personal:path', ABS_PATH),
               ('personal:first-person', FIRST_PERSON), ('personal:first-person', FIRST_PERSON_WE),
               ('personal:first-person', FIRST_PERSON_I), ('personal:quote', QUOTE),
               ('personal:history', HISTORY), ('personal:hardware', HARDWARE),
               ('personal:private-tool', PRIVATE_TOOL))

# ---- fragment shapes ---------------------------------------------------------------------
FRAGMENTS = (
    ('empty parens', re.compile(r'(?<![\w\]\)\'"])\(\s*\)')),
    ('dangling see', re.compile(r'\bsee\s+[.,;:)]')),
    ('space before full stop', re.compile(r'[A-Za-z0-9)`\'"]\s+\.(?=\s|$)')),
    ('double comma', re.compile(r',\s*,')),
    ('empty aside', re.compile(r'\(\s*(?:since|from|on|in|see|e\.g\.|i\.e\.|as of)\s*\)', re.I)),
    ('empty e.g.', re.compile(r'\b(?:e\.g\.|i\.e\.)\s*[.,;)]')),
    ('cut subject', re.compile(r'\b(?:[Ii]f|[Ww]hen|where|until|after|before)\s+(?:wants|wanted|put\s+it|chose|picked|noticed|decided|tried)\b')),
    ('dangling function word', re.compile(r'\b(?:than|the|an)\s*[.;:,](?=\s|$)')),
    ('cut punctuation', re.compile(r'(?<![\'"\[(]):,(?![\'"]|\s+\w+:|[\d.}]|d\b)|,\s*[.;](?=\s|$)')),
    ('doubled word', re.compile(r'\b(on|the|of|to|in|a|is|and)\s+\1\b')),
    ('article swap', re.compile(r'(?<![+*/=-]\s)(?<![\w.\'-])a\s+(?!one\b|once\b|eu|ewe|o\')(?=[aeio])[a-z]{3,}\b')),
)
LEADING_PUNCT = re.compile(r'^\s*[,;](?!\s*$)|^\s*:\s')              # a comment opening on cut punctuation
DANGLING_END = re.compile(r'\s(?:the|a|an|of|to|and|or|by|with|for|from|at|its|their)\s*$', re.I)
SPLIT_PARENS = re.compile(r'\(\s*\n[\s#]*\)')
SECTION_HDR = re.compile(r'^\s*#\s*[-=]{3,}\s*\S.*?[-=]{3,}\s*#?\s*$')

# ---- cited files -------------------------------------------------------------------------
REPO_DIRS = ('lib', 'stages', 'pipeline', 'fetch', 'docs', 'tests', 'areas', 'assets', 'tools', '.githooks')
# top-level directories of the build's own output: a path under one is run-time, not cited
RUNTIME_DIRS = ('work', 'data', 'logs', 'geom', 'ortho', 'facades', 'publish', 'scratch', 'build', 'out',
                'tiles', 'panos', 'context', 'lidar', 'osm', 'mapillary', 'trees', 'props', 'objects',
                'masks', 'textures', 'atlas')
# files the build writes at run time (area.yaml is also committed per area)
RUNTIME_FILES = {'derived.yaml', 'area.yaml', 'manifest.json', 'source.json', 'SOURCES.json',
                 'ATTRIBUTION.txt', 'FROZEN.md'}
CITED = re.compile(r'(?<![\w/.${}<>*-])((?:[\w*<>.-]+/)*[\w*<>-][\w*<>.-]*\.(?:py|md|sh|yaml|yml|json|toml))\b')
PLACEHOLDER = re.compile(r'<[^>]*>|\bNAME\b|\bAREA\b|NN|(?<=_)name\b')   # <name>, NAME, sNN_*.py
STAGE_REF = re.compile(r'(?<![\w/.-])(?:s(\d{2}[a-z]?)(?![\w.])|[Ss]tages?\s+(\d{2}[a-z]?)\b)'
                       r'|(?<![\w/.-])(?:s|[Ss]tages?\s+)\d{2}[a-z]?\s*(?:-|\.\.|to|/)\s*s?(\d{2}[a-z]?)\b')
STAGE_MOD = re.compile(r'(?<![\w/.-])(s(?:\d{2}[a-z]?|2x)_[a-z]\w*)')
URL = re.compile(r'\bhttps?://\S+')

# ---- allowlist: (path glob, rule prefix, substring of the hit line, reason) --------------
ALLOW = (
    ('stages/s10a_3dfier.py', 'personal:path', '/data/', 'mount points inside the 3dfier container'),
    ('tests/test_area_cli.py', 'cited', '_start_', 'marker files the test itself writes'),
    ('tests/test_area_cli.py', 'cited', 's99_nope', 'a stage name the test expects to be refused'),
)
# files whose string literals are the gate's own patterns or its test samples
SELF_STRINGS = ('tools/check_public.py', 'tests/test_public_gate.py')
NEXT_LEAD = re.compile(r'^\s*(?:#+|//|\*|-)?\s*')


def tracked(root):
    try:
        out = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                             cwd=root, capture_output=True, check=True).stdout
        names = sorted({n for n in out.decode().split('\0') if n})
    except (OSError, subprocess.CalledProcessError):
        names = []
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in ('.git', '.venv', '__pycache__')
                      and not (dp == root and d in ('data', 'work', 'logs'))]
            names += [os.path.relpath(os.path.join(dp, f), root) for f in fns]
    return [n for n in names if os.path.isfile(os.path.join(root, n))]


def load_denylist(root, path=None):
    p = path or os.environ.get('OCF_PRIVATE_DENYLIST') or os.path.join(root, '.private-denylist')
    if not os.path.isfile(p):
        return None, p
    rx = [re.compile(l.strip(), re.I) for l in open(p, encoding='utf-8', errors='replace')
          if l.strip() and not l.strip().startswith('#')]
    return rx, p


def is_text(path):
    if os.path.splitext(path)[1].lower() in BINARY_EXT:
        return False
    with open(path, 'rb') as f:
        return b'\0' not in f.read(8192)


def prose_of_python(text):
    """(line, text, kind) for every comment and string literal; kind 'comment' or 'string'."""
    out = []
    kinds = {tokenize.STRING} | {getattr(tokenize, 'FSTRING_MIDDLE', tokenize.STRING)}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out.append((tok.start[0], tok.string[1:], 'comment'))
            elif tok.type in kinds:
                out += [(tok.start[0] + i, part, 'string') for i, part in enumerate(tok.string.split('\n'))]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out = [(i, l, 'comment') for i, l in enumerate(text.split('\n'), 1)]
    return out


def cited_ok(name, files, basenames):
    if '/' in name:
        first = name.split('/', 1)[0]
        if first == '.':
            return True
        if first not in REPO_DIRS:                       # run-time output, unless it is code
            return not name.endswith(('.py', '.sh')) or first in RUNTIME_DIRS
        pat = PLACEHOLDER.sub('*', name)
        return any(fnmatch.fnmatch(f, pat) for f in files) or any(f.startswith(name.rstrip('/') + '/') for f in files)
    if name in RUNTIME_FILES or not name.endswith(('.py', '.md', '.sh')):
        return True
    pat = PLACEHOLDER.sub('*', name)
    return any(fnmatch.fnmatch(b, pat) for b in basenames)


def check_prose(rel, ln, s, hits, ends_block=False, kind='line'):
    """The prose rules on one line of prose."""
    lic = LICENCE_LINE.search(s)
    s = URL.sub('<url>', s)
    for name, rx in PROSE_RULES:
        m = rx.search(s)
        if m and not (lic and name in ('personal:date', 'personal:year')):
            hits.append((rel, ln, name, m.group(0)))
    for name, rx in FRAGMENTS:
        m = rx.search(s)
        if m:
            hits.append((rel, ln, 'fragment:' + name, m.group(0)))
    if kind == 'comment' and LEADING_PUNCT.search(s):
        hits.append((rel, ln, 'fragment:leading punctuation', s.strip()[:30]))
    if ends_block and s.strip() and DANGLING_END.search(s):
        hits.append((rel, ln, 'fragment:dangling last word', s.strip()[-30:]))


def scan_binaries(root, files, deny):
    """A .glb's JSON chunk gets the address shapes and the denylist; an image must carry no
    EXIF / XMP / IPTC segment or PNG text chunk."""
    import json
    hits = []
    for rel in files:
        ext = os.path.splitext(rel)[1].lower()
        try:
            b = open(os.path.join(root, rel), 'rb').read()
        except OSError:
            continue
        if ext == '.glb' and b[:4] == b'glTF' and len(b) >= 20:
            js = b[20:20 + struct.unpack('<I', b[12:16])[0]].decode('utf-8', 'replace')
            try:
                js = json.dumps(json.loads(js.rstrip(' \0')), indent=0)
            except ValueError:
                pass
            for i, line in enumerate(js.split('\n'), 1):
                for name, rx in LINE_RULES + (('personal:date', DATE),):
                    m = rx.search(line)
                    if m:
                        hits.append((rel, i, 'binary:' + name, m.group(0)))
                hits += [(rel, i, 'binary:denylist', m.group(0)) for rx in deny or () for m in [rx.search(line)] if m]
        elif ext in ('.jpg', '.jpeg'):
            k = 2
            while k + 4 <= len(b) and b[k] == 0xFF and b[k + 1] != 0xDA:
                if b[k + 1] in (0xE1, 0xFE, 0xED):
                    hits.append((rel, 1, 'binary:image metadata', 'APP1/COM/IPTC segment'))
                    break
                k += 2 + struct.unpack('>H', b[k + 2:k + 4])[0]
        elif ext == '.png' and b[:8] == b'\x89PNG\r\n\x1a\n':
            k = 8
            while k + 8 <= len(b):
                ln, typ = struct.unpack('>I', b[k:k + 4])[0], b[k + 4:k + 8]
                if typ in (b'tEXt', b'iTXt', b'zTXt', b'eXIf'):
                    hits.append((rel, 1, 'binary:image metadata', typ.decode()))
                    break
                k += 12 + ln
    return hits


def scan_commits(root, revs, deny):
    """Author, committer, dates and message of every commit reachable from `revs`."""
    hits = []
    fmt = '%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%ad%x1f%cd%x1f%B%x1e'
    out = subprocess.run(['git', 'log', '--date=iso-strict', '--format=' + fmt, *revs], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    for rec in out.split('\x1e'):
        f = rec.strip('\n').split('\x1f')
        if len(f) < 8:
            continue
        sha, meta, msg = 'commit ' + f[0][:10], ' '.join(f[1:5]), f[7]
        hits += [(sha, 0, 'commit:non-UTC date', d) for d in (f[5], f[6]) if not d.endswith(('+00:00', 'Z'))]
        hits += [(sha, 0, 'commit:denylist', m.group(0)) for rx in deny or () for t in (meta, msg)
                 for m in [rx.search(t)] if m]
        text = meta.replace('noreply' + '@example.invalid', '') + ' ' + msg
        hits += [(sha, 0, 'commit:' + n.split(':')[1], m.group(0)) for n, rx in LINE_RULES for m in [rx.search(text)] if m]
        for i, line in enumerate(msg.split('\n'), 1):
            check_prose(sha, i, line, hits)
    return hits


def scan_file(rel, text, deny, files, basenames, stages, modules, hits):
    lines = text.split('\n')
    for ln, line in enumerate(lines, 1):                          # every line, code included
        hits += [(rel, ln, name, m.group(0)) for name, rx in LINE_RULES for m in [rx.search(line)] if m]
        if rel not in SELF_STRINGS:
            hits += [(rel, ln, name, m.group(0)) for name, rx in CODE_RULES for m in [rx.search(line)] if m]
        hits += [(rel, ln, 'denylist', m.group(0)) for rx in deny or () for m in [rx.search(line)] if m]
        if deny and ln < len(lines):                              # a name split across a line break
            a = line.rstrip()
            j = a + ' ' + NEXT_LEAD.sub('', lines[ln])
            hits += [(rel, ln, 'denylist', m.group(0)) for rx in deny for m in rx.finditer(j)
                     if m.start() < len(a) < m.end()]
    for m in SPLIT_PARENS.finditer(text):
        hits.append((rel, text.count('\n', 0, m.start()) + 1, 'fragment:empty parens', '( )'))
    for i, l in enumerate(lines if not rel.endswith('.md') else ()):
        if SECTION_HDR.match(l):
            j = next((k for k in range(i + 1, len(lines)) if lines[k].strip()), len(lines))
            if j == len(lines) or SECTION_HDR.match(lines[j]):
                hits.append((rel, i + 1, 'fragment:empty section', l.strip()[:40]))
    python = rel.endswith('.py') or text.startswith('#!') and 'python' in text.split('\n', 1)[0]
    prose = prose_of_python(text) if python else [(i, l, 'line') for i, l in enumerate(lines, 1)]
    for k, (ln, s, kind) in enumerate(prose):
        if rel in SELF_STRINGS and kind == 'string':
            continue                                              # the gate's own patterns and samples
        nxt = prose[k + 1] if k + 1 < len(prose) else None
        if kind == 'line':
            ends = nxt is None or not nxt[1].strip()
        else:
            ends = kind == 'comment' and (nxt is None or nxt[2] != 'comment' or nxt[0] != ln + 1)
        check_prose(rel, ln, s, hits, ends, kind)
        if nxt is not None and nxt[0] == ln + 1 and not (rel in SELF_STRINGS and nxt[2] == 'string'):
            a = URL.sub('<url>', s.rstrip())                      # a history phrase split across a line break
            j = a + ' ' + NEXT_LEAD.sub('', URL.sub('<url>', nxt[1]))
            hits += [(rel, ln, 'personal:history', m.group(0)) for m in HISTORY.finditer(j)
                     if m.start() < len(a) < m.end()]
        for m in CITED.finditer(s):
            if not cited_ok(m.group(1), files, basenames):
                hits.append((rel, ln, 'cited:missing file', m.group(1)))
        for m in STAGE_REF.finditer(s):
            n = m.group(1) or m.group(2) or m.group(3)
            if n not in stages and n.rstrip('abcdefgh') not in stages:
                hits.append((rel, ln, 'cited:missing stage', m.group(0)))
        for m in STAGE_MOD.finditer(s):
            num = re.match(r's(\d{2}[a-z]?)_', m.group(1))
            if (num.group(1) not in stages) if num else (m.group(1) not in modules):
                hits.append((rel, ln, 'cited:missing module', m.group(1)))


def scan(root, quiet=False, require_denylist=False, commits=None, deny_path=None):
    files = tracked(root)
    deny, deny_path = load_denylist(root, deny_path)
    if deny is None and require_denylist:
        return [('(denylist)', 0, 'denylist:missing', 'no denylist at %s; set OCF_PRIVATE_DENYLIST' % deny_path)]
    if deny is None and not quiet:
        print('check_public: no denylist at %s; the denylist rule is skipped' % deny_path, file=sys.stderr)
    basenames = {os.path.basename(f) for f in files}
    stages = {m.group(1) for f in files if f.startswith('stages/')
              for m in [re.match(r's(\d{2}[a-z]?)_', os.path.basename(f))] if m}
    modules = {os.path.splitext(os.path.basename(f))[0] for f in files if f.endswith('.py')}
    hits = scan_binaries(root, files, deny)
    for rel in files:
        if is_text(os.path.join(root, rel)):
            text = open(os.path.join(root, rel), encoding='utf-8', errors='replace').read()
            scan_file(rel, text, deny, files, basenames, stages, modules, hits)
    if commits:
        hits += scan_commits(root, commits, deny)
    out = []
    for rel, ln, rule, txt in hits:
        p = os.path.join(root, rel)
        if os.path.isfile(p) and not rule.startswith('binary:'):
            line = (open(p, encoding='utf-8', errors='replace').read().split('\n') + [''])[ln - 1]
            if any(fnmatch.fnmatch(rel, g) and rule.startswith(r) and sub in line for g, r, sub, _ in ALLOW):
                continue
        out.append((rel, ln, rule, txt))
    return sorted(set(out), key=lambda h: (h[0], h[1], h[2]))


def scan_trees(root, revs, quiet=False, require_denylist=False):
    """Every rule on the tree of each commit reachable from `revs` (one scan per distinct tree).
    Hits are reported as `<commit>:<path>`."""
    import shutil, tarfile, tempfile
    _, deny_path = load_denylist(root)
    shas = subprocess.run(['git', 'rev-list', *revs], cwd=root, capture_output=True, text=True,
                          check=True).stdout.split()
    seen, hits = set(), []
    for sha in shas:
        tree = subprocess.run(['git', 'rev-parse', sha + '^{tree}'], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        if tree in seen:
            continue
        seen.add(tree)
        tmp = tempfile.mkdtemp(prefix='check_public_tree_')
        try:
            arc = subprocess.run(['git', 'archive', '--format=tar', sha], cwd=root, capture_output=True,
                                 check=True).stdout
            with tarfile.open(fileobj=io.BytesIO(arc)) as tf:
                tf.extractall(tmp, filter='data') if hasattr(tarfile, 'data_filter') else tf.extractall(tmp)
            hits += [('%s:%s' % (sha[:10], rel), ln, rule, txt)
                     for rel, ln, rule, txt in scan(tmp, True, require_denylist, None, deny_path)]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--require-denylist', action='store_true',
                    help='fail (instead of skipping the rule) when the denylist file is absent')
    ap.add_argument('--commits', nargs='*', metavar='REV',
                    help='also check author, committer, dates (must be UTC) and message of these '
                         'revisions and their history (default with no REV: HEAD)')
    ap.add_argument('--trees', nargs='*', metavar='REV',
                    help='also run every rule on the committed file tree of these revisions and '
                         'their history (default with no REV: HEAD)')
    a = ap.parse_args(argv)
    hits = scan(a.root, a.quiet, a.require_denylist, None if a.commits is None else (a.commits or ['HEAD']))
    if a.trees is not None:
        hits += scan_trees(a.root, a.trees or ['HEAD'], a.quiet, a.require_denylist)
    for rel, ln, rule, txt in hits:
        print('%s:%d: %s: %s' % (rel, ln, rule, txt.strip()[:120]))
    if hits:
        print('check_public: %d hit(s); fix each sentence at its source' % len(hits), file=sys.stderr)
        return 1
    if not a.quiet:
        print('check_public: clean')
    return 0


if __name__ == '__main__':
    sys.exit(main())
