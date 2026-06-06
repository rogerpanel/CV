"""v21: aggressive content condensation to hit ≤170 pages with council
typography (14pt Times Roman, 1.5 spacing, council margins).

Strategy per chapter:
- Ch 1: Compress every subsection body to 2-3 sentences max
- Ch 2: Replace all proof bodies with one-line refs to Appendix A;
  compress motivation paragraphs around theorems
- Ch 3: Light compression only (preserve software/programming details
  per user directive: "Chapter 3 must be built with software packages
  and programming done for the dissertation for reproducibility")
- Ch 4: Compress experimental writeups to results-only
- Ch 5: Drop platform-internals sections (already done in v18, reapply)
- Ch 6: Compress to UAV essentials
- Introduction: Compress to skeleton (relevance, aim, tasks, novelty,
  defended provisions only)
"""
import re, os

RU_DIR = '/home/user/CV/dissertation_v12/chapters'
EN_DIR = '/home/user/CV/dissertation_v12_eng/chapters'


def read(p): return open(p).read()
def write(p, s): open(p, 'w').write(s)


def shrink_proof_bodies(s):
    """Replace long \begin{proof}...\end{proof} bodies with one-line refs."""
    def repl(m):
        body = m.group(1)
        n_lines = body.count('\n')
        if n_lines < 4:
            return m.group(0)  # short proof - keep
        # Long proof - replace
        return ('\\begin{proof} Полное доказательство приведено в '
                'Приложении~А. \\end{proof}')
    s = re.sub(r'\\begin{proof}(.*?)\\end{proof}', repl, s, flags=re.DOTALL)
    return s


def shrink_proof_bodies_en(s):
    def repl(m):
        body = m.group(1)
        if body.count('\n') < 4:
            return m.group(0)
        return ('\\begin{proof} The full proof is given in '
                'Appendix~A. \\end{proof}')
    return re.sub(r'\\begin{proof}(.*?)\\end{proof}', repl, s, flags=re.DOTALL)


def remove_long_paragraphs_between(s, prefix_kept_chars=300):
    """Between every two structural elements (\section/\subsection/
    \begin{...}), keep only the FIRST `prefix_kept_chars` characters
    of pure-prose paragraph content."""
    # Structural separators
    sep = re.compile(r'(\n\\(?:section|subsection|subsubsection|chapter|paragraph)\{)')
    parts = sep.split(s)
    out = [parts[0]]
    for i in range(1, len(parts), 2):
        sep_token = parts[i]
        body_and_rest = parts[i+1] if i+1 < len(parts) else ''
        # Find end of the section heading (closing })
        m = re.match(r'([^}]*\}\s*(?:\\label\{[^}]+\}\s*)?(?:%%=*\s*\n)?\s*)', body_and_rest, re.DOTALL)
        if not m:
            out.append(sep_token + body_and_rest)
            continue
        heading = m.group(1)
        body = body_and_rest[len(heading):]
        # Find end of this section's pure-prose paragraph (until next
        # \section/\subsection/\subsubsection/\begin/\paragraph/$/\[/\noindent\textbf)
        end_match = re.search(
            r'(\n\\(?:section|subsection|subsubsection|chapter|paragraph)\{|\\begin\{|\\\[|\\noindent\\textbf|\n%%=)',
            body)
        if end_match:
            prose = body[:end_match.start()]
            tail = body[end_match.start():]
        else:
            prose = body
            tail = ''
        # Truncate prose to first 1-2 sentences
        sentences = re.split(r'(?<=[.!?])\s+', prose.strip())
        kept = ' '.join(sentences[:2]).strip()
        if kept and not kept.endswith('.'):
            kept += '.'
        new_section = sep_token + heading + ('\n' + kept + '\n\n' if kept else '\n')
        out.append(new_section + tail)
    return ''.join(out)


def compress_chapter(path, aggressive=True, preserve_software=False, en=False):
    s = read(path)
    orig = s.count('\n')
    if not preserve_software:
        # Aggressive: keep only 1-2 sentences per section body
        s = remove_long_paragraphs_between(s)
    # Always: collapse blank lines
    s = re.sub(r'\n\n\n+', '\n\n', s)
    new = s.count('\n')
    write(path, s)
    return orig, new


def compress_ch2(path, en=False):
    """Ch 2: shrink proof bodies + general compression."""
    s = read(path)
    orig = s.count('\n')
    if en:
        s = shrink_proof_bodies_en(s)
    else:
        s = shrink_proof_bodies(s)
    # Also compress motivation text
    s = remove_long_paragraphs_between(s)
    s = re.sub(r'\n\n\n+', '\n\n', s)
    new = s.count('\n')
    write(path, s)
    return orig, new


def compress_ch5(path, en=False):
    """Ch 5: nuke all sections not in supervisor TOC 5.1-5.7."""
    s = read(path)
    orig = s.count('\n')
    # Targets to remove (regardless of position)
    if en:
        kill = [
            'Platform Overview and Target Users',
            'System Architecture',
            'Functional Groups and Page Inventory',
            'Functional Groups and Page Registry',
            'AI Command Centre',
            'Detection Models',
            'SOC Intelligence Suite',
            'LLM Security Testing',
            'MLSecOps Standards Compliance',
            'MLSecOps Standards',
            'Multi-Agent PQC-IDS Module',
            'Multi-Agent PQC-IDS',
            'Edge-Agent Rust Migration and Fast XDP Path',
            'Edge-Agent Rust Migration',
            'Operational Deployment Infrastructure',
            'Open-Source Contribution',
        ]
    else:
        kill = [
            'Обзор платформы и целевые пользователи',
            'Архитектура системы',
            'Функциональные группы и реестр страниц',
            'Командный центр ИИ',
            'Модели обнаружения',
            'Пакет SOC-аналитики',
            'Тестирование безопасности LLM',
            'Соответствие стандартам MLSecOps',
            'Модуль Multi-Agent PQC-IDS',
            'Миграция граничного агента на Rust и быстрый путь XDP',
            'Операционная инфраструктура развёртывания',
            'Вклад в открытый исходный код',
        ]
    for title in kill:
        pattern = re.compile(
            r'(%%=*\s*\n)?\\section\{' + re.escape(title) + r'\}.*?(?=\n%%=*\s*\n\\section\{|\n\\section\{|\n\\chapter|\Z)',
            re.DOTALL)
        s, n = pattern.subn('', s, count=1)
    # Also nuke the chapter intro (long preamble)
    s = re.sub(
        r'(\\chapter\{[^}]+\}\s*\\label\{[^}]+\})\s*\n[^\\]{200,3000}(?=\n%%=|\n\\section)',
        r'\1\n\n', s, flags=re.DOTALL)
    s = re.sub(r'\n\n\n+', '\n\n', s)
    new = s.count('\n')
    write(path, s)
    return orig, new


def compress_intro(path, en=False):
    """Introduction: compress to skeleton."""
    s = read(path)
    orig = s.count('\n')
    s = remove_long_paragraphs_between(s)
    s = re.sub(r'\n\n\n+', '\n\n', s)
    new = s.count('\n')
    write(path, s)
    return orig, new


# Process all chapters
targets = [
    # (filename, function, kwargs)
    ('chapter1_v9_RU.tex', compress_chapter, {'aggressive': True}),
    ('chapter2_v9_RU.tex', compress_ch2, {}),
    ('chapter3_v9_RU.tex', compress_chapter, {'aggressive': False, 'preserve_software': True}),
    ('chapter4_v9_RU.tex', compress_chapter, {'aggressive': True}),
    ('chapter5_v9_RU.tex', compress_ch5, {}),
    ('chapter6_v11_RU.tex', compress_chapter, {'aggressive': True}),
    ('Introduction_v10a_Rus.tex', compress_intro, {}),
]
en_targets = [
    ('chapter1_v9.tex', compress_chapter, {'aggressive': True, 'en': True}),
    ('chapter2_v9.tex', compress_ch2, {'en': True}),
    ('chapter3_v9.tex', compress_chapter, {'aggressive': False, 'preserve_software': True, 'en': True}),
    ('chapter4_v9.tex', compress_chapter, {'aggressive': True, 'en': True}),
    ('chapter5_v9.tex', compress_ch5, {'en': True}),
    ('chapter6_v11.tex', compress_chapter, {'aggressive': True, 'en': True}),
    ('Introduction_v10.tex', compress_intro, {'en': True}),
]

total = 0
for fname, fn, kwargs in targets:
    p = os.path.join(RU_DIR, fname)
    o, n = fn(p, **kwargs)
    if o != n:
        print(f'RU {fname}: {o} -> {n} (-{o-n})')
    total += (o-n)
for fname, fn, kwargs in en_targets:
    p = os.path.join(EN_DIR, fname)
    if not os.path.exists(p):
        print(f'SKIP {p}: not found')
        continue
    o, n = fn(p, **kwargs)
    if o != n:
        print(f'EN {fname}: {o} -> {n} (-{o-n})')
    total += (o-n)
print(f'TOTAL: {total}')
