# Общая компонента всех направлений отказа: выученные реперы + DIM

Популяция: 12 выученных реперов (theft k=5, illegal_activities k=5, malicious_use k=5, all k=7, сиды [21, 3, 7]) + 4 DIM как одномерные реперы. hidden = 2048.
Нуль — 300 наборов случайных ортонормированных реперов тех же форм и весов.

Веса: **pooled** — 1/N по всем объектам; **balanced** — половина веса конусам, половина DIM. Пол lam_1 = max_j c_j, потолок 1.0.

Конусная общая компонента (шаг 2) лежит отдельно в `results/common_axis/global/pooled.pt` и здесь используется только для сверки.

## Спектр смешанной популяции

| DIM в точке | веса | N | пол | lam_1 | нуль (среднее / p95) | lam_2 | cos(w_1, w_1 конусов) |
|---|---|---|---|---|---|---|---|
| native | pooled | 16 | 0.062 | **0.545** | 0.085 / 0.087 | 0.363 | 0.978 |
| native | balanced | 16 | 0.125 | **0.524** | 0.133 / 0.137 | 0.273 | 0.862 |
| pos-4_L22 | pooled | 16 | 0.062 | **0.574** | 0.085 / 0.087 | 0.350 | 0.960 |
| pos-4_L22 | balanced | 16 | 0.125 | **0.635** | 0.133 / 0.137 | 0.234 | 0.777 |
| pos-1_L27 | pooled | 16 | 0.062 | **0.553** | 0.085 / 0.087 | 0.373 | 0.923 |
| pos-1_L27 | balanced | 16 | 0.125 | **0.622** | 0.133 / 0.136 | 0.265 | 0.659 |

## Кто несёт ось: DIM в точке native, веса balanced

Конусы: alpha от 0.621 до 0.754 (среднее 0.692). DIM: theft 0.734, illegal_activities 0.735, malicious_use 0.777, all 0.765.

| объект | alpha = \|B w_1\| |
|---|---|
| theft/seed_21 (k=5) | 0.710 |
| theft/seed_3 (k=5) | 0.660 |
| theft/seed_7 (k=5) | 0.627 |
| illegal_activities/seed_21 (k=5) | 0.655 |
| illegal_activities/seed_3 (k=5) | 0.709 |
| illegal_activities/seed_7 (k=5) | 0.697 |
| malicious_use/seed_21 (k=5) | 0.754 |
| malicious_use/seed_3 (k=5) | 0.621 |
| malicious_use/seed_7 (k=5) | 0.683 |
| all/seed_21 (k=7) | 0.706 |
| all/seed_3 (k=7) | 0.737 |
| all/seed_7 (k=7) | 0.751 |
| **DIM theft** | **0.734** |
| **DIM illegal_activities** | **0.735** |
| **DIM malicious_use** | **0.777** |
| **DIM all** | **0.765** |

## Кто несёт ось: DIM в точке pos-4_L22, веса balanced

Конусы: alpha от 0.486 до 0.734 (среднее 0.617). DIM: theft 0.932, illegal_activities 0.947, malicious_use 0.946, all 0.934.

| объект | alpha = \|B w_1\| |
|---|---|
| theft/seed_21 (k=5) | 0.579 |
| theft/seed_3 (k=5) | 0.522 |
| theft/seed_7 (k=5) | 0.486 |
| illegal_activities/seed_21 (k=5) | 0.533 |
| illegal_activities/seed_3 (k=5) | 0.624 |
| illegal_activities/seed_7 (k=5) | 0.566 |
| malicious_use/seed_21 (k=5) | 0.716 |
| malicious_use/seed_3 (k=5) | 0.603 |
| malicious_use/seed_7 (k=5) | 0.635 |
| all/seed_21 (k=7) | 0.687 |
| all/seed_3 (k=7) | 0.718 |
| all/seed_7 (k=7) | 0.734 |
| **DIM theft** | **0.932** |
| **DIM illegal_activities** | **0.947** |
| **DIM malicious_use** | **0.946** |
| **DIM all** | **0.934** |

## Кто несёт ось: DIM в точке pos-1_L27, веса balanced

Конусы: alpha от 0.461 до 0.668 (среднее 0.582). DIM: theft 0.947, illegal_activities 0.951, malicious_use 0.951, all 0.944.

| объект | alpha = \|B w_1\| |
|---|---|
| theft/seed_21 (k=5) | 0.663 |
| theft/seed_3 (k=5) | 0.626 |
| theft/seed_7 (k=5) | 0.602 |
| illegal_activities/seed_21 (k=5) | 0.633 |
| illegal_activities/seed_3 (k=5) | 0.638 |
| illegal_activities/seed_7 (k=5) | 0.668 |
| malicious_use/seed_21 (k=5) | 0.632 |
| malicious_use/seed_3 (k=5) | 0.461 |
| malicious_use/seed_7 (k=5) | 0.539 |
| all/seed_21 (k=7) | 0.490 |
| all/seed_3 (k=7) | 0.512 |
| all/seed_7 (k=7) | 0.524 |
| **DIM theft** | **0.947** |
| **DIM illegal_activities** | **0.951** |
| **DIM malicious_use** | **0.951** |
| **DIM all** | **0.944** |

Смешанные оси сохранены: `/home/jovyan/f.zakharov/geometry-of-refusal/results/common_axis/global/mixed_<точка>_<веса>.pt`. Конусная ось шага 2 (`pooled.pt`) не перезаписывается.

