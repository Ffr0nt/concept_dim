# Concept cones в geometry-of-refusal

Где и как устроены «конусы отказа». Пути относительно `geometry-of-refusal/`.
Основной фокус для проекта concept_dim.

## Идея
Одиночное «направление отказа» — один вектор в скрытом пространстве. **Концепт-конус**
обобщает его до **низкоразмерного подпространства** (обычно 2–3 измерения): любое
направление «из конуса» тоже опосредует отказ. Конус задаётся **ортонормированным
базисом** `fn_vectors`.

## Где задаётся

| Место | Что |
|---|---|
| `rdo.py:555` | **`class RefusalCone(nn.Module)`** — сам конус (базис + вмешательства) |
| `rdo.py:643` | **`refusal_cone_optimization(...)`** — обучение базиса конуса |
| `rdo.py:1029` | `train_refusal_cone(group_name, run_name, init_vectors, **kwargs)` — точка входа |
| `rdo.py:491/498` | `sample_hypersphere_gaussian` / `sample_prob_vectors` — сэмплинг направлений из конуса |
| `refusal_direction/pipeline/submodules/select_direction.py:509` | `select_cone_basis(...)` — отбор/оценка базиса конуса в пайплайне |
| `refusal_direction/pipeline/run_rdo_pipeline.py:77,181` | использование `select_cone_basis`; `is_subspace = cone_dim > 1` |

Конфиг/флаги (`rdo.py`): `--train_cone`, `--min_cone_dim`/`--max_cone_dim` (дефолт 2–3),
`--n_sample`, `--sampling_method` (`hypersphere`/`interpolation`), `--optimize_basis`.

## RefusalCone — разбор (rdo.py:555-640)

- **Базис (обучаемые параметры)** — `rdo.py:560`:
  ```python
  self.fn_vectors = [nn.Parameter(torch.randn(dim), requires_grad=True) for _ in range(n_vectors)]
  ```
  `n_vectors` = размерность конуса. Можно инициализировать `init_vectors` (нормируются).

- **`transform(sample)`** (`rdo.py:600`) — из коэффициентов делает направление **внутри**
  конуса: `sample @ stack(fn_vectors)`, затем нормировка. Так из базиса получают конкретные
  направления конуса для сэмплирования.

- **`__call__(direction)`** (`rdo.py:568`) — **аблация** направления по ВСЕМ слоям:
  вход слоя, выход self-attn, выход mlp:
  ```python
  for layer in self.module.layers:
      self.ablate_input(layer, d)                 # layer.input  -= proj(layer.input, d)
      self.ablate_output(layer.self_attn, d, 3)   # выход attn — кортеж длины 3
      self.ablate_output(layer.mlp, d, 1)         # выход mlp
  ```
  Аблация = `activation - projection_einops(activation, direction)` (`rdo.py:576-593`).

- **`add(direction, alpha, layer_idx)`** (`rdo.py:595`) — activation addition в конкретный слой.

- **`orthogonalize()`** (`rdo.py:609`) — Gram–Schmidt по базису (ортонормировка) + (опция)
  проекция первого вектора в **нуль-пространство** `orthogonal_vectors` (`rdo.py:616-636`):
  `P = Aᵀ(AAᵀ)⁻¹A`, `v_ortho = (I−P)v` — так конус делают **независимым** от заданных
  направлений (representational independence).

- **`parameters()`** (`rdo.py:606`) — отдаёт `fn_vectors` оптимизатору.

## Как обучается (refusal_cone_optimization, rdo.py:643)

1. `operation = RefusalCone(model.model, hidden_size, cone_dim, …)` (`rdo.py:665`).
2. `optimizer = AdamW(operation.parameters(), …)` (`rdo.py:667`) — учатся только `fn_vectors`.
3. Сэмплятся направления из конуса: `sample_hypersphere_gaussian(n_sample, cone_dim)`
   → `transform` → направление (`rdo.py:702,720`).
4. Три лосса на каждом сэмпле (и на базисных векторах), делятся на `cone_dim`:
   - **ablation** (CE) — `rdo.py:731-733`;
   - **addition** (CE) — `rdo.py:741-743`;
   - **retain** (KL к базовым логитам) — `rdo.py:753-755`.
   Все `.backward()` **внутри** `with model.trace()` ⇒ градиент течёт сквозь nnsight-вмешательства к базису.
   Веса λ: `ablation_lambda=1`, `addition_lambda=0.2`, `retain_lambda=1` (`DEFAULT_CONFIG`).

Логическая цепочка:
`RefusalCone` (базис + аблация) → `transform`/сэмплинг (направления из конуса)
→ `__call__` (аблация по слоям) → 3 лосса → backprop к `fn_vectors`.

## ⚠️ Что мешает прямому переиспользованию в concept_dim
- `RefusalCone` хардкодит **`.cuda()`** (`rdo.py:560,564`) — привязка к GPU.
- Использует **глобальную** переменную `model` (`rdo.py:570,602`) вместо переданной явно.
- `__call__` завязан на структуру `module.layers[*].self_attn/.mlp` (архитектура decoder-слоя).

Кандидат на аккуратный рефактор для concept_dim: вынести `device`/`model` в аргументы,
убрать глобали и `.cuda()`, чтобы `RefusalCone` можно было импортировать и использовать
как самостоятельный компонент.

## Что импортировать
Для работы с конусами в concept_dim, скорее всего, понадобятся:
```python
from rdo import RefusalCone, refusal_cone_optimization, train_refusal_cone
from rdo import sample_hypersphere_gaussian, sample_prob_vectors
from generate_utils import projection_einops
```
(Помнить: `import rdo` на верхнем уровне грузит дефолтную модель — см. `geometry-of-refusal-map.md`.)
