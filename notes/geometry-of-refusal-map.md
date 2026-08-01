# geometry-of-refusal — карта кода

Заметки о том, **как и где** расположен код в библиотеке `geometry-of-refusal`
(форк: https://github.com/Ffr0nt/geometry-of-refusal, на сервере — `~/f.zakharov/geometry-of-refusal`).
Пути ниже даны относительно корня репозитория `geometry-of-refusal/`.

Репозиторий к работе **«The Geometry of Refusal in LLMs»** (Wollschläger et al.):
изучение направления и «конуса» отказа в residual stream выровненных (chat) моделей.
Метод — **RDO (Refusal Direction Optimization)**, надстроенный поверх пайплайна
refusal_direction (Arditi et al.).

## Раскладка

```
geometry-of-refusal/
├── rdo.py                       # ЯДРО: RDO + класс RefusalCone (концепт-конус)
├── scoring.py                   # метрики отказа / bypass / induce
├── generate_utils.py            # проекции, генерация, вмешательства fn-вектором
├── plots.py, plot_style.py      # графики и стиль
├── cosinesim_analysis.py        # анализ косинусной близости
├── monotonic_scaling_property.py# проверка monotonic scaling property
├── requirements.txt
├── .env_example                 # какие env-переменные нужны (см. ниже)
├── data/                        # generate_datasets.py, saladbench_splits
└── refusal_direction/           # ВЛОЖЕННЫЙ пайплайн (Arditi), отдельный source-root
    ├── pipeline/
    │   ├── config.py
    │   ├── model_utils/          # обёртки моделей: gemma / llama3 / qwen + factory
    │   ├── submodules/           # generate_directions, select_direction, evaluate_*
    │   ├── utils/hook_utils.py   # PyTorch forward-hooks (ablation/addition)
    │   ├── run_pipeline.py
    │   ├── run_rdo_pipeline.py   # использует select_cone_basis
    │   └── run_rdo_samples.py
    └── dataset/                  # load_dataset.py + harmful/harmless split-ы (json)
```

## Две кодовые базы и импорты

- **Корневой кластер** (`rdo`, `scoring`, `generate_utils`, `plots`, …) — импортируют друг
  друга как top-level (`from generate_utils import …`). Требуют **корень репо** на `sys.path`.
  Это то, что установлено как editable-пакет (см. `pyproject.toml` форка).
- **`refusal_direction/`** — отдельный source-root: код внутри пишет `from pipeline…`,
  `from dataset…`, т.е. ждёт саму `refusal_direction/` на пути. Запускается как скрипты
  (`python refusal_direction/pipeline/run_*.py`), НЕ ставится как пакет.

## Корневые модули — где что

### rdo.py — ядро (главные ориентиры)

| Строка | Объект | Назначение |
|---|---|---|
| 27  | `DEFAULT_CONFIG` | все дефолты (модель `google/gemma-2-2b-it`, lr, cone_dim, λ-веса лоссов) |
| 66  | `parse_args()` | CLI-аргументы; при импорте возвращает дефолты (стр. 74-75) |
| 147-164 | код верхнего уровня | **при импорте** сразу грузит `LanguageModel(MODEL_PATH)` — модель тянется на импорте |
| 165 | `model.requires_grad_(False)` | веса модели заморожены |
| 223 | `apply_chat_template()` | chat-шаблоны: ветки llama-3 / gemma / qwen2.5 (224-231) |
| 239-300 | `generate_*` | генерация первого токена, harmful/harmless таргетов |
| 376 | `build_prompts_and_labels()` | сборка промптов и меток для 3 лоссов |
| 412 | `CustomDataset` / 477 `custom_collate` | датасет/коллейт |
| 491 | `sample_hypersphere_gaussian()` | сэмплинг направлений на гиперсфере (для конуса) |
| 498 | `sample_prob_vectors()` | альтернативный сэмплинг (interpolation) |
| 503 | `compute_ce_loss()` / 516 `kl_div_fn()` | лоссы (CE для ablation/addition, KL для retain) |
| 527 | `get_cosine_sims_for_vector()` | косинусные близости активаций к вектору |
| **555** | **`RefusalCone(nn.Module)`** | **концепт-конус** — см. `concept-cones.md` |
| **643** | **`refusal_cone_optimization()`** | **обучение конуса/направления** (осн. цикл) |
| 952 | `train_refusal_vector()` | обучить одиночное направление (cone_dim=1) |
| 1029 | `train_refusal_cone()` | обучить конус (n>1 базисных векторов) — точка входа |
| 1106 | `DirectionalAblation(nn.Module)` | аблация фиксированного направления |
| 1184 | `repind_rdo()` | representational independence RDO |
| 1405 | `train_independent_vector()` | обучить направление, независимое от заданных |

### scoring.py — метрики
- `refusal_score` (21), `refusal_metric` (37) — метрика отказа по логитам;
- `get_logits` (51), `get_bypass_scores` (71), `get_induce_scores` (93) — оценка
  «обхода» отказа (bypass) и «наведения» отказа (induce), с опциональным fn-вектором.

### generate_utils.py — операции над активациями
- `projection_einops` (7) — проекция активации на направление (база всех аблаций);
- `generate_completions` (16) — генерация с вмешательствами;
- `intervene_with_fn_vector_ablation` (36) / `..._addition` (68) — вмешательства
  fn-вектором (ablation/addition) через nnsight.

### Прочие корневые
- `plots.py` — построение всех графиков (`plot_scores`, `plot_safety_scores`,
  `create_latex_table`, `load_evaluations` …);
- `plot_style.py` — `apply_style()` (единый стиль matplotlib);
- `cosinesim_analysis.py` — `new_get_cosine_similarities()` (косинус активаций до/после
  вмешательства);
- `monotonic_scaling_property.py` — проверка монотонности эффекта при масштабировании.

## refusal_direction/ — вложенный пайплайн

- `pipeline/config.py` — конфиг пайплайна (`Config`).
- `pipeline/model_utils/` — обёртки под конкретные модели: `gemma_model.py`,
  `llama3_model.py`, `qwen_model.py`, базовый `model_base.py` (`ModelBase`),
  `model_factory.py` (`construct_model_base`).
- `pipeline/submodules/`:
  - `generate_directions.py` — `get_mean_activations`, `get_mean_diff`,
    `generate_directions` (DIM — difference-in-means направление);
  - `select_direction.py` — `select_direction` (150), `select_rdo_direction` (325),
    **`select_cone_basis` (509)** — отбор направления/базиса конуса; `get_refusal_scores` (35);
  - `evaluate_jailbreak.py` — `evaluate_jailbreak` (substring-judge и др.);
  - `evaluate_loss.py` — расчёт лоссов по датасетам (alpaca/pile/chat).
- `pipeline/utils/hook_utils.py` — PyTorch forward-hooks
  (`get_activation_addition_input_pre_hook`, `get_all_direction_ablation_hooks`).
  ⚠️ Здесь вмешательства через **hooks**, а в `rdo.py` — через **nnsight** (два разных механизма).
- `pipeline/run_rdo_pipeline.py` — прогон RDO-пайплайна (`is_subspace = cone_dim > 1`).
- `dataset/load_dataset.py` + json-split-ы (`harmful/harmless_{train,val,test}.json`).

## Ключевые механизмы (пояснения)

### nnsight (в rdo.py) — чтение/вмешательство/градиенты
- `with model.trace(...)` — контекст трейсинга; `layer.input` / `layer.output` — активации
  (Envoy-прокси). `.save()` — вытащить значение наружу.
- **Вмешательство** — присваивание: `layer.output = new_activation`, `layer.input -= projection`.
- Вмешательства встроены в **autograd-граф** ⇒ `.backward()` внутри трейса пускает
  градиент сквозь них к обучаемым векторам (веса модели заморожены). Так и учится
  направление/конус.
- nnsight умеет и **прямое вмешательство в градиенты** через `.grad` (в этом репо не
  используется, но механизм есть).
- Требует `torch>=2.4` ⇒ не ставится на Intel-macOS (поэтому окружение только на сервере).

### Модель по умолчанию
`google/gemma-2-2b-it` — только **дефолт** (`DEFAULT_CONFIG['model']`), меняется флагом
`--model`. Выбрана как маленькая (2B) **instruction-tuned** модель: отказ — свойство
выровненных моделей, а 2B дёшева для многих RDO-прогонов. Поддержаны семейства
gemma / llama-3 / qwen2.5. Модель **gated** на HF ⇒ для запуска нужен HF-токен с доступом.

### Concept cones
Вынесено в отдельную заметку: `concept-cones.md`.

## Окружение (.env)
Из `.env_example` (создать `.env` в корне geometry-of-refusal):
- `HUGGINGFACE_CACHE_DIR` — кэш HF (веса);
- `SAVE_DIR` — куда писать результаты (по умолчанию `./results`);
- `DIM_DIR` — папка DIM-направлений (`dim`);
- `WANDB_ENTITY`, `WANDB_PROJECT` — wandb (в нашем workflow **не используем**, политика).
