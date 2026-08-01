# SALAD-Bench — таксономия и распределение данных

Источник: **OpenSafetyLab/Salad-Data** (HF), проект SALAD-Bench.
Все числа посчитаны по сабсету **`base_set`** (`train`, **21 318** строк) —
загрузка `load_dataset("OpenSafetyLab/Salad-Data", name="base_set")`.

## Сабсеты датасета (всего ~30 658)
| Сабсет | Строк | Что |
|---|---:|---|
| `base_set` | 21 318 | базовые вредные вопросы (здесь считаем таксономию) |
| `attack_enhanced_set` | 5 000 | + методы джейлбрейка |
| `mcq_set` | 3 840 | вопросы с вариантами (multiple-choice) |
| `defense_enhanced_set` | 200 | усиленные защитой версии |
| `small_attack_subset` | 300 | компактная выборка для быстрых прогонов |

## Колонки `base_set`
`qid`, `question`, `source`, `1-category`, `2-category`, `3-category`.
Таксономия **иерархическая, 3 уровня**: **6 доменов → 16 задач → 66 листьев**.

## Уровень 1 — домены (6)
| Строк | % | Домен |
|---:|---:|---|
| 8 756 | 41.1% | O5: Malicious Use |
| 6 486 | 30.4% | O1: Representation & Toxicity |
| 2 031 | 9.5% | O2: Misinformation Harms |
| 1 717 | 8.1% | O6: Human Autonomy & Integrity |
| 1 477 | 6.9% | O4: Information & Safety |
| 851 | 4.0% | O3: Socioeconomic Harms |

## Статистика по листьям (3-category, 66 шт.)
- Диапазон: **min 86, max 964**, медиана **292**, среднее **323**.
- Порог по числу примеров в листе: `>1000: 0`, `>500: 8`, `>300: 31`, `>100: 65`, `>50: 66`.
- Ни одного пустого или «гигантского» листа; дисбаланс в основном на уровне доменов/задач.

## Полное дерево (домен — задача — лист, с числом примеров)

Формат: `домен — всего` / `задача — всего (число листьев)` / `лист — число`.

### O5: Malicious Use — 8756
- **O14: Illegal Activities** — 3422 (8 листьев)
    - O57: Theft — 964
    - O56: Violent Crimes — 759
    - O53: Financial Crimes — 358
    - O54: Drug-related Crimes — 351
    - O55: Sexual Offenses — 296
    - O59: Environmental Crimes — 248
    - O58: Illegal Law Advice — 224
    - O60: Traffic and Driving Offenses — 222
- **O12: Fraud or Deceptive Action** — 2439 (8 листьев)
    - O44: Academic Dishonesty — 393
    - O43: Financial Fraud — 367
    - O42: Scams — 349
    - O45: Fake Review Generation — 292
    - O46: Fake Online Content — 280
    - O47: Disinformation — 271
    - O49: Pseudo-pharmaceuticals — 257
    - O48: Spam — 230
- **O10: Security Threats** — 1643 (5 листьев)
    - O35: Weapon Generation and Transportation — 462
    - O38: Cyber Attack — 332
    - O37: Malware Generation — 326
    - O36: Management or Operation of Critical Infrastructure — 322
    - O39: Biological and Chemical Harms — 201
- **O13: Influence Operations** — 815 (3 листьев)
    - O52: Illegitimate Surveillance — 380
    - O50: Manipulation of Public Opinion — 246
    - O51: Campaign Materials — 189
- **O11: Defamation** — 437 (2 листьев)
    - O41: Defamation Someone — 269
    - O40: Group Defamation — 168

### O1: Representation & Toxicity — 6486
- **O1: Toxic Content** — 3246 (7 листьев)
    - O7: Insult — 643
    - O2: Harass, Threaten, or Bully An Individual — 550
    - O1: Hate Speech — 530
    - O5: Violent Content — 511
    - O3: Enjoying someone else's Pain — 368
    - O4: Terrorism — 330
    - O6: Child Abuse — 314
- **O2: Unfair Representation** — 2165 (6 листьев)
    - O8: Racial and Ethnic Stereotyping — 847
    - O11: Gender Stereotyping — 367
    - O12: Religious Stereotyping — 301
    - O13: Location Stereotyping — 243
    - O9: Age Stereotyping — 222
    - O10: Disability Stereotyping — 185
- **O3: Adult Content** — 1075 (3 листьев)
    - O14: Erotic Chat — 378
    - O15: Sexual Activity — 361
    - O16: Pornography — 336

### O2: Misinformation Harms — 2031
- **O4: Erosion of Trust in Public Information** — 1314 (4 листьев)
    - O19: Illegal Drugs and Regulated/Controlled Substances — 563
    - O18: Common Sense — 320
    - O17: Law Advice — 244
    - O20: Medical Advice — 187
- **O5: Propagating Misconceptions/False Beliefs** — 717 (3 листьев)
    - O21: Historical Revisionism — 293
    - O23: Radicalization — 229
    - O22: Science Denialism — 195

### O6: Human Autonomy & Integrity — 1717
- **O16: Violation of Personal Property** — 1077 (4 листьев)
    - O65: Plagiarism — 324
    - O67: Forge Identity — 320
    - O66: Copyright Infringement — 238
    - O64: Mimicking Writing Style — 195
- **O15: Persuasion and Manipulation** — 640 (2 листьев)
    - O62: Self-Harm — 388
    - O63: Psychological Manipulation — 252

### O4: Information & Safety — 1477
- **O8: Dissemination of Dangerous Information** — 820 (4 листьев)
    - O30: Social Media Rumors — 253
    - O32: Financial Data Leaks — 243
    - O29: Fake News — 184
    - O31: Government Secrets — 140
- **O9: Privacy Infringement** — 657 (2 листьев)
    - O33: Inferring Personal Information — 373
    - O34: Leaking Private Information — 284

### O3: Socioeconomic Harms — 851
- **O6: Risky Financial Practices** — 651 (3 листьев)
    - O26: Gambling — 261
    - O24: Multi-level Marketing — 230
    - O25: Paypal Lending — 160
- **O7: Trade and Compliance** — 200 (2 листьев)
    - O27: Labor Issues — 114
    - O28: Trade Affairs — 86

## Связь с проектом
- В форке `geometry-of-refusal/data/saladbench_splits` лежат только `instruction` + `source`
  (harmful/harmless выборки для RDO), **без** колонок таксономии. Чтобы привязать
  концепты/направления к категориям, таксономию из `base_set` надо джойнить самому
  (по тексту вопроса) либо строить свои сплиты прямо из `base_set`.
- Дисбаланс важен при выборе гранулярности концептов: 1 конус на домен (6) —
  максимально сбалансировано; на задачу (16) — умеренно; на лист (66) — 86–964 примера/лист.
