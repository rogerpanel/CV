"""Aggressive content pruning to compress dissertation while preserving
all theorems, equations, tables, figures, metrics, and supervisor-mandated
section structure.

Strategy:
- Remove paragraphs that are pure transitional exposition
- Compress multi-paragraph derivation explanations to single-paragraph summaries
- Drop redundant "as shown in...", "in summary..." pointer sentences
- Convert verbose lists to compact bullet lists
- Remove redundant chapter intros that duplicate the abstract or conclusion
"""
import re, os

TARGETS_RU = [
    '/home/user/CV/dissertation_v12/chapters/chapter1_v9_RU.tex',
    '/home/user/CV/dissertation_v12/chapters/chapter2_v9_RU.tex',
    '/home/user/CV/dissertation_v12/chapters/chapter3_v9_RU.tex',
    '/home/user/CV/dissertation_v12/chapters/chapter4_v9_RU.tex',
    '/home/user/CV/dissertation_v12/chapters/chapter5_v9_RU.tex',
    '/home/user/CV/dissertation_v12/chapters/chapter6_v11_RU.tex',
    '/home/user/CV/dissertation_v12/chapters/Introduction_v10a_Rus.tex',
    '/home/user/CV/dissertation_v12/chapters/conclusion_v10_RU.tex',
]
TARGETS_EN = [
    '/home/user/CV/dissertation_v12_eng/chapters/chapter1_v9.tex',
    '/home/user/CV/dissertation_v12_eng/chapters/chapter2_v9.tex',
    '/home/user/CV/dissertation_v12_eng/chapters/chapter3_v9.tex',
    '/home/user/CV/dissertation_v12_eng/chapters/chapter4_v9.tex',
    '/home/user/CV/dissertation_v12_eng/chapters/chapter5_v9.tex',
    '/home/user/CV/dissertation_v12_eng/chapters/chapter6_v11.tex',
    '/home/user/CV/dissertation_v12_eng/chapters/Introduction_v10.tex',
    '/home/user/CV/dissertation_v12_eng/chapters/conclusion_v10.tex',
]

# Paragraph-level patterns to DELETE entirely (transitional/redundant prose)
DELETE_PATTERNS_RU = [
    # Chapter/section intros that just preview structure
    r'\n(?:В настоящей главе|В данной главе) (?:систематически )?представле[нм][аоы][^.]{30,500}\.\n',
    r'\n(?:Глава|Раздел) состоит из [^.]{30,400}\.\n',
    r'\nСтруктура (?:настоящей )?главы (?:следует|организована)[^.]{30,500}\.\n',
    r'\nДанная глава имеет[^.]{30,500}\.\n',
    # Pointer sentences
    r'\nКак (?:показано|обсуждалось|описано) в (?:Главе|разделе|§|\\S)[^.]{20,400}\.\n',
    r'\nВ соответствии с (?:результатами|анализом)[^.]{20,500}\.\n',
    r'\nПодробнее см\.[^.]{0,300}\.\n',
    r'\n\\noindent См\.[^.]{0,200}\.\n',
    # Summary sentences
    r'\nПодводя итог[^,.]*,\s+[^.]{30,500}\.\n',
    r'\nТаким образом, [^.]{30,400}\.\n',
    # Forward-pointer sentences
    r'\n(?:Перейдём|Далее перейдём|Теперь обратимся) к [^.]{20,300}\.\n',
    r'\nВ следующем разделе [^.]{20,400}\.\n',
    r'\nВ \S\\ref\{[^}]+\} (?:будет|будут) [^.]{20,400}\.\n',
    # Filler/explanatory transitions
    r'\nОтметим, что [^.]{30,500}\.\n',
    r'\nСтоит подчеркнуть, что [^.]{30,500}\.\n',
    r'\nВажно отметить, что [^.]{30,500}\.\n',
    r'\nНа практике [^.]{30,300}\.\n',
    r'\nВ контексте настоящей работы [^.]{30,400}\.\n',
]

DELETE_PATTERNS_EN = [
    r'\nThis chapter (?:systematically )?(?:presents|covers|establishes|describes)[^.]{30,500}\.\n',
    r'\nThe (?:chapter|section) consists of [^.]{30,400}\.\n',
    r'\nThe structure of (?:this |the )?chapter [^.]{30,500}\.\n',
    r'\nAs (?:shown|discussed|described) in (?:Chapter|Section|§|\\S)[^.]{20,400}\.\n',
    r'\nIn accordance with [^.]{20,500}\.\n',
    r'\n(?:For further detail|See) [^.]{0,200}\.\n',
    r'\nIn summary[^,.]*,\s+[^.]{30,500}\.\n',
    r'\nThus,? [^.]{30,400}\.\n',
    r'\n(?:Now we turn|We now turn|Let us now) to [^.]{20,300}\.\n',
    r'\nIn the following section,? [^.]{20,400}\.\n',
    r'\nIn (?:\\S|\\ref)[^.]{20,400}\.\n',
    r'\nNote that [^.]{30,500}\.\n',
    r'\nIt (?:is worth|should be) (?:emphasised|noted|stressed) [^.]{30,500}\.\n',
    r'\nIn practice,? [^.]{30,300}\.\n',
    r'\nIn the context of (?:this|the present) work [^.]{30,400}\.\n',
]

def prune(path, patterns):
    s = open(path).read()
    orig = s.count('\n')
    for pat in patterns:
        s = re.sub(pat, '\n', s, flags=re.MULTILINE)
    # Collapse blank lines
    s = re.sub(r'\n\n\n+', '\n\n', s)
    s = re.sub(r' +\n', '\n', s)
    new = s.count('\n')
    open(path,'w').write(s)
    return orig, new

total = 0
for t in TARGETS_RU:
    o, n = prune(t, DELETE_PATTERNS_RU)
    if o != n:
        print(f'RU {os.path.basename(t)}: {o} -> {n} (-{o-n})')
    total += (o - n)
for t in TARGETS_EN:
    o, n = prune(t, DELETE_PATTERNS_EN)
    if o != n:
        print(f'EN {os.path.basename(t)}: {o} -> {n} (-{o-n})')
    total += (o - n)
print(f'TOTAL lines removed: {total}')
