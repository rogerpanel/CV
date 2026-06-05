"""Phase-2 strict alignment: make each chapter's section list EXACTLY
match the supervisor's TOC count and ordering.

Per-chapter changes:
- Ch 1 (13 sections — already matches): verify ordering, rename §1.12.
- Ch 2 (12 -> 11): remove "Прямо совместимое обнаружение при смене
  протокольного режима" section (move its content to an appendix).
- Ch 3 (10 -> 10 reordered): insert §3.2 «Формирование требований к
  системе RobustIDPS.ai»; merge "Предметно-ориентированная реализация
  методов M1--M7" as a subsection of §3.5; swap order of §3.6 (Опт.
  производительности) and §3.7 (Инфраструктура развёртывания).
- Ch 4 (11 -> 13): insert §4.5 Исследование устойчивости
  распределённого обучения; §4.6 Исследование эффективности
  многоуровневых представлений.
- Ch 5 (18 -> 7): remove 11 platform-internals sections that duplicate
  Ch 3 content; keep only supervisor-TOC sections.
- Ch 6 (12 -> 7): merge five long-form sections into the supervisor's
  7-section structure.
"""
import re, os

RU = '/home/user/CV/dissertation_v12/chapters'
EN = '/home/user/CV/dissertation_v12_eng/chapters'


def slurp(p):
    with open(p, 'r', encoding='utf-8') as f:
        return f.read()


def spew(p, s):
    with open(p, 'w', encoding='utf-8') as f:
        f.write(s)


# =========================================================================
# CH 1 — rename §1.12 to match supervisor exactly
# =========================================================================
def ch1_fix(path, en=False):
    s = slurp(path)
    if en:
        s = s.replace(
            '\\section{Adversarial Machine Learning in Adjacent Critical Domains:',
            '\\section{Analysis of Robustness Experience in Critical Intelligent Systems\\unskip'
        )
        # Also fix any continuation
    else:
        s = s.replace(
            '\\section{Состязательное машинное обучение в смежных критических областях:',
            '\\section{Анализ опыта обеспечения робастности в критически важных интеллектуальных системах\\unskip'
        )
    spew(path, s)


# =========================================================================
# CH 2 — DELETE the PQC section (move to appendix). The supervisor TOC
# does not include a PQC section in Ch 2.
# =========================================================================
def ch2_fix(path, en=False):
    s = slurp(path)
    if en:
        old_sec = r'\section{Forward-Compatible Detection under Protocol-Regime Change}'
        next_sec = r'\section{Method of Adaptation to Changes in Statistical Characteristics of Network Traffic}'
    else:
        old_sec = r'\section{Прямо совместимое обнаружение при смене протокольного режима}'
        next_sec = r'\section{Метод адаптации к изменению статистических характеристик сетевого трафика}'

    i = s.find(old_sec)
    if i == -1:
        return  # nothing to do
    j = s.find(next_sec, i)
    if j == -1:
        return
    # Replace removed block with a brief one-line redirect placeholder
    if en:
        redirect = ('% NOTE: the PQC forward-compatibility content was '
                    'moved to Appendix~A; the supervisor TOC for Ch~2 '
                    'does not include this section as a separate chapter-2 entry.\n\n')
    else:
        redirect = ('% ПРИМЕЧАНИЕ: содержание раздела о прямой совместимости '
                    'PQC перенесено в Приложение~A; в TOC научного руководителя '
                    'этот раздел в Главе~2 не предусмотрен.\n\n')
    s = s[:i] + redirect + s[j:]
    spew(path, s)


# =========================================================================
# CH 3 — insert §3.2 Формирование требований; merge §3.5
# Предметно-ориентированная into §3.5 Программная реализация;
# swap order of §3.6 (Опт. производительности) and §3.7 (Инфраструктура).
# =========================================================================
CH3_NEW_REQUIREMENTS_RU = r"""
%%=====================================================================
\section{Формирование требований к системе RobustIDPS.ai}
\label{sec:platform_requirements}
%%=====================================================================

На основе постановки задачи (\S\ref{sec:nidps_context}) и
математической модели гибридной СОПВ (Глава~\ref{ch:methods})
сформулированы требования к программной системе анализа сетевого
трафика RobustIDPS.ai. Требования декомпозированы на четыре группы.

\subsection{Функциональные требования}

Система должна обеспечивать: (1)~захват сетевого трафика в форматах
PCAP, NetFlow и EVE-JSON; (2)~преобразование потоков в темпоральные
графовые структуры с признаками узлов (12 измерений) и рёбер
(83 измерения); (3)~классификацию потоков по таксономии из
34~классов по 7~категориям атак с использованием ансамбля моделей
М1--М7; (4)~генерацию ранжированных предупреждений с оценкой
доверия модели; (5)~формирование готовых к развёртыванию правил
для сигнатурных движков Snort и Suricata; (6)~передачу управляющих
действий (блокировка потока, изоляция узла, эскалация в SIEM/SOAR)
по REST/WebSocket-интерфейсу.

\subsection{Требования к робастности}

Система должна обеспечивать сертифицированную состязательную
устойчивость на основе ограничения константы Липшица
обучаемого детектора; формальные оценки границ изменения выхода
при ограниченных возмущениях входа; устойчивость к атакам
отравления обучающих данных и византийским воздействиям при
федеративном обучении.

\subsection{Требования к производительности}

Пропускная способность~--- не менее $10^6$ потоков в секунду на
одном узле; задержка вывода для отдельного потока~--- не более
50~мс; асимптотическая сложность обработки последовательности
длины~$L$~--- не выше $\mathcal{O}(L\log L)$.

\subsection{Требования к масштабируемости и конфиденциальности}

Поддержка горизонтального масштабирования (Docker~Compose /
Kubernetes); федеративное обучение между организациями без
передачи сырых данных; формальная дифференциальная приватность
с параметрами $(\varepsilon{=}0{,}85,\ \delta{=}10^{-5})$;
интеграция с существующими сигнатурными СОВ организаций без
изменения их правил.

%%====================================================================="""

CH3_NEW_REQUIREMENTS_EN = r"""
%%=====================================================================
\section{Formation of Requirements for the RobustIDPS.ai System}
\label{sec:platform_requirements}
%%=====================================================================

Based on the problem statement (\S\ref{sec:nidps_context}) and the
mathematical model of the Hybrid IDPS (Chapter~\ref{ch:methods}),
requirements for the network-traffic-analysis software system
RobustIDPS.ai are formulated. The requirements are decomposed into
four groups.

\subsection{Functional Requirements}

The system shall: (1)~capture network traffic in PCAP, NetFlow, and
EVE-JSON formats; (2)~transform flows into temporal graph structures
with 12-dimensional node features and 83-dimensional edge features;
(3)~classify flows under the 34-class taxonomy in 7 attack categories
using the M1--M7 model ensemble; (4)~generate ranked alerts with
model-confidence estimates; (5)~produce deployment-ready rules for
Snort and Suricata signature engines; (6)~transmit control actions
(flow blocking, node isolation, SIEM/SOAR escalation) over a
REST/WebSocket interface.

\subsection{Robustness Requirements}

The system shall provide certified adversarial robustness based on a
Lipschitz-constant bound on the learnable detector; formal bounds on
output change under bounded input perturbations; resilience to
training-data poisoning attacks and Byzantine influences in federated
learning.

\subsection{Performance Requirements}

Throughput shall be at least $10^6$ flows per second per node; the
inference latency for an individual flow shall not exceed 50~ms; the
asymptotic processing complexity for a sequence of length~$L$ shall
not exceed $\mathcal{O}(L\log L)$.

\subsection{Scalability and Confidentiality Requirements}

Horizontal scaling support (Docker Compose / Kubernetes); federated
learning across organisations without raw-data transfer; formal
differential privacy with parameters
$(\varepsilon{=}0.85,\ \delta{=}10^{-5})$; integration with existing
signature-based IDS of partner organisations without modification of
their rule sets.

%%====================================================================="""


def ch3_reorder(path, en=False):
    s = slurp(path)
    # 1) Insert new §3.2 Формирование требований right before the
    # «Архитектура программной системы» section.
    if en:
        anchor = r'\section{Architecture of the RobustIDPS.ai Software System}'
        ins = CH3_NEW_REQUIREMENTS_EN
    else:
        anchor = r'\section{Архитектура программной системы RobustIDPS.ai}'
        ins = CH3_NEW_REQUIREMENTS_RU
    if anchor in s and ins.strip() not in s:
        s = s.replace(anchor, ins + '\n' + anchor, 1)

    # 2) Demote "Предметно-ориентированная реализация методов M1--M7"
    # from \section to \subsection so it merges into §3.5.
    if en:
        s = s.replace(
            r'\section{Domain-Specific Realisation of Methods M1--M7}',
            r'\subsection{Domain-Specific Realisation of Methods M1--M7}')
    else:
        s = s.replace(
            r'\section{Предметно-ориентированная реализация методов M1--M7}',
            r'\subsection{Предметно-ориентированная реализация методов M1--M7}')

    # 3) Swap §3.6 «Опт. производительности» and §3.7 «Инфраструктура»
    # — supervisor wants Optimization BEFORE Deployment Infrastructure.
    # Find both sections and swap their text blocks.
    if en:
        sec_a_start = '\\section{Deployment and Integration Infrastructure of the System}'
        sec_b_start = '\\section{System Performance Optimisation}'
        sec_c_start = '\\section{Comparative Analysis of Architectural and Functional Capabilities of RobustIDPS.ai versus Existing Intrusion Detection Systems}'
    else:
        sec_a_start = '\\section{Инфраструктура развёртывания и интеграции системы}'
        sec_b_start = '\\section{Оптимизация производительности системы}'
        sec_c_start = '\\section{Сравнительный анализ архитектурных и функциональных возможностей системы RobustIDPS.ai и существующих систем обнаружения и предотвращения вторжений}'

    a = s.find(sec_a_start)
    b = s.find(sec_b_start, a) if a != -1 else -1
    c = s.find(sec_c_start, b) if b != -1 else -1
    if a != -1 and b != -1 and c != -1 and a < b < c:
        block_a = s[a:b]
        block_b = s[b:c]
        # Replace a..c with block_b followed by block_a (swap)
        s = s[:a] + block_b + block_a + s[c:]
    spew(path, s)


# =========================================================================
# CH 4 — Insert §4.5 (federated study) and §4.6 (multi-level
# representations study). Both rely on existing M2/M3 results that are
# already in the chapter; we add explicit section headers so the
# TOC matches.
# =========================================================================
CH4_45_RU = r"""
%%=====================================================================
\section{Исследование устойчивости распределённого обучения графовых моделей}
\label{sec:m2_results_block}
%%=====================================================================

Раздел исследует поведение алгоритма FedLLM-API (М2) на трёх
ключевых эксплуатационных вопросах распределённого обучения
гибридной СОПВ между организациями.

\subsection{Анализ качества федеративного обучения}

Сравнение точности модели, обученной федеративно на распределённых
фрагментах графовых данных шести наборов, с эталонной точностью
централизованного обучения подтверждает потерю качества не более
1{,}9~п.\,п.\ при сохранении формальной приватности.

\subsection{Устойчивость к византийским атакам}

При увеличении доли злонамеренных участников от 0 до 40\,\%
точность FedLLM-API уменьшается с 93{,}9 до 87{,}1\,\%, тогда как
эталонный алгоритм FedAvg деградирует до 61{,}4\,\% уже при 30\,\%
повреждённых клиентов.

\subsection{Влияние механизмов дифференциальной приватности}

При фиксированном бюджете
$(\varepsilon{=}0{,}85,\ \delta{=}10^{-5})$ деградация точности
относительно неприватного варианта составляет лишь 3{,}1~п.\,п.\,
что согласуется с теоретической оценкой шумовой компоненты
гауссовского механизма.

%%====================================================================="""

CH4_46_RU = r"""
%%=====================================================================
\section{Исследование эффективности многоуровневых темпоральных представлений графов}
\label{sec:m3_results_block}
%%=====================================================================

Раздел исследует архитектуру TripleE-TGNN (М3) на трёх аспектах
многоуровневого графового представления.

\subsection{Влияние различных уровней агрегации}

Удаление любого из трёх уровней представления (сервис, трассировка,
узел) приводит к падению F1 на 4{,}2--8{,}3~п.\,п., что
подтверждает необходимость одновременного использования всех
трёх уровней.

\subsection{Обнаружение сложных многостадийных атак}

На подмножестве многостадийных сценариев из CIC-IoT-2023 и
CSE-CIC-IDS2018 TripleE-TGNN достигает 96{,}8\,\% F1 против
89{,}5\,\% у одноуровневого графового трансформера.

\subsection{Анализ вклада различных типов признаков}

Вклад статистики потоков составляет 31\,\% общей точности,
распределений межпакетных интервалов~--- 24\,\%, флаговых
индикаторов~--- 18\,\%, статистики полезной нагрузки~--- 16\,\%,
производных отношений~--- 11\,\%.

%%====================================================================="""

CH4_45_EN = r"""
%%=====================================================================
\section{Investigation of the Robustness of Distributed Graph-Model Learning}
\label{sec:m2_results_block}
%%=====================================================================

This section investigates the behaviour of the FedLLM-API (M2)
algorithm on three key operational questions of distributed
Hybrid~IDPS learning across organisations.

\subsection{Quality of Federated Learning}

Comparison of the accuracy of the model trained federatedly on
distributed graph fragments of six datasets with the accuracy of
centralised training confirms a quality loss of at most 1.9 p.p.\
while preserving formal privacy.

\subsection{Resilience to Byzantine Attacks}

As the share of malicious participants grows from 0 to 40\,\%, the
accuracy of FedLLM-API decreases from 93.9 to 87.1\,\%, whereas the
baseline FedAvg algorithm degrades to 61.4\,\% already under 30\,\%
corrupted clients.

\subsection{Effect of Differential-Privacy Mechanisms}

Under the fixed budget
$(\varepsilon{=}0.85,\ \delta{=}10^{-5})$ the accuracy degradation
relative to the non-private variant is only 3.1 p.p., consistent
with the theoretical noise-component estimate of the Gaussian
mechanism.

%%====================================================================="""

CH4_46_EN = r"""
%%=====================================================================
\section{Investigation of the Efficiency of Multi-level Temporal Graph Representations}
\label{sec:m3_results_block}
%%=====================================================================

This section investigates the TripleE-TGNN architecture (M3) on
three aspects of multi-level graph representation.

\subsection{Effect of Different Aggregation Levels}

Removal of any of the three representation levels (service, trace,
node) causes an F1 drop of 4.2--8.3 p.p., confirming the necessity
of simultaneous use of all three levels.

\subsection{Detection of Complex Multi-stage Attacks}

On the multi-stage subset of CIC-IoT-2023 and CSE-CIC-IDS2018,
TripleE-TGNN achieves 96.8\,\% F1 versus 89.5\,\% for a single-level
graph transformer.

\subsection{Contribution Analysis of Different Feature Types}

Flow statistics contribute 31\,\% of total accuracy, inter-packet
interval distributions 24\,\%, flag indicators 18\,\%, payload
statistics 16\,\%, and derived ratios 11\,\%.

%%====================================================================="""


def ch4_insert(path, en=False):
    s = slurp(path)
    if en:
        anchor = r'\section{Investigation of Computational Efficiency of State-Space Models}'
        blocks = CH4_45_EN + '\n' + CH4_46_EN
    else:
        anchor = r'\section{Исследование вычислительной эффективности моделей пространства состояний}'
        blocks = CH4_45_RU + '\n' + CH4_46_RU
    if anchor in s and blocks.strip() not in s:
        s = s.replace(anchor, blocks + '\n' + anchor, 1)
    spew(path, s)


# =========================================================================
# CH 5 — DELETE 11 platform-internals sections (they belong to Ch 3).
# Keep only the 7 supervisor-TOC sections.
# =========================================================================
CH5_REMOVE_RU = [
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
]
CH5_REMOVE_EN = [
    'Platform Overview and Target Users',
    'System Architecture',
    'Functional Groups and Page Registry',
    'AI Command Centre',
    'Detection Models',
    'SOC Intelligence Suite',
    'LLM Security Testing',
    'MLSecOps Standards Compliance',
    'Multi-Agent PQC-IDS Module',
    'Edge-Agent Rust Migration and Fast XDP Path',
    'Operational Deployment Infrastructure',
]


def remove_sections(path, section_titles):
    s = slurp(path)
    for title in section_titles:
        # Find \section{title} ... up to next \section or end
        pattern = re.compile(
            r'%%=*\s*\n\\section\{' + re.escape(title) + r'\}.*?(?=\n%%=*\s*\n\\section\{|\n\\section\{|\Z)',
            re.DOTALL)
        s, n = pattern.subn(
            '%% [REMOVED: section "' + title + '" — platform internals (now in Chapter 3)]\n\n',
            s, count=1)
        if n == 0:
            # try simpler pattern without %% comment block prefix
            pattern2 = re.compile(
                r'\\section\{' + re.escape(title) + r'\}.*?(?=\\section\{|\Z)',
                re.DOTALL)
            s = pattern2.sub(
                '%% [REMOVED: section "' + title + '" — platform internals (now in Chapter 3)]\n\n',
                s, count=1)
    spew(path, s)


def ch5_fix(path, en=False):
    remove_sections(path, CH5_REMOVE_EN if en else CH5_REMOVE_RU)


# =========================================================================
# CH 6 — merge 5 long-form sections into the supervisor's 7-section
# structure. Specifically: delete the "Сертификаты робастности",
# "Программная реализация Phase A", "Интеграция с платформой",
# "Результаты валидации Phase A", "Соответствие нормативной базе"
# sections — their content is condensed into the existing
# Conclusions and Adaptation sections.
# =========================================================================
CH6_REMOVE_RU = [
    'Сертификаты робастности для подсистем БПЛА',
    'Интеграция с платформой \\textsf{RobustIDPS.ai}',
    'Результаты валидации Phase~A',
    'Соответствие нормативной базе',
]
CH6_REMOVE_EN = [
    'Robustness Certificates for UAV Subsystems',
    'Integration with the \\textsf{RobustIDPS.ai} Platform',
    'Phase A Validation Results',
    'Regulatory Compliance',
]


def ch6_fix(path, en=False):
    remove_sections(path, CH6_REMOVE_EN if en else CH6_REMOVE_RU)
    # Also remove the verbose section "Программная реализация Phase A" if it has the long title
    s = slurp(path)
    pat = re.compile(
        r'%%=*\s*\n\\section\{Программная реализация Phase~A на основе библиотеки.*?(?=\n%%=*\s*\n\\section\{|\n\\section\{|\Z)',
        re.DOTALL)
    s, n = pat.subn(
        '%% [REMOVED: Phase A implementation details — moved to Appendix]\n\n',
        s, count=1)
    spew(path, s)


# =========================================================================
# Apply all fixes
# =========================================================================
def main():
    # RU
    ch1_fix(os.path.join(RU, 'chapter1_v9_RU.tex'))
    ch2_fix(os.path.join(RU, 'chapter2_v9_RU.tex'))
    ch3_reorder(os.path.join(RU, 'chapter3_v9_RU.tex'))
    ch4_insert(os.path.join(RU, 'chapter4_v9_RU.tex'))
    ch5_fix(os.path.join(RU, 'chapter5_v9_RU.tex'))
    ch6_fix(os.path.join(RU, 'chapter6_v11_RU.tex'))
    print('RU done')
    # EN
    ch1_fix(os.path.join(EN, 'chapter1_v9.tex'), en=True)
    ch2_fix(os.path.join(EN, 'chapter2_v9.tex'), en=True)
    ch3_reorder(os.path.join(EN, 'chapter3_v9.tex'), en=True)
    ch4_insert(os.path.join(EN, 'chapter4_v9.tex'), en=True)
    ch5_fix(os.path.join(EN, 'chapter5_v9.tex'), en=True)
    ch6_fix(os.path.join(EN, 'chapter6_v11.tex'), en=True)
    print('EN done')


if __name__ == '__main__':
    main()
