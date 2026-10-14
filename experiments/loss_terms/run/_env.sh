# Общее окружение раннеров loss_terms (source из run/*.sh, сам по себе не запускается).
# Скрипты идут в окружении shared-refusal-component (`uv run` из его корня): оттуда пакет
# shared_refusal и форк geometry-of-refusal (editable). Данные — его базовый запуск.
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

HERE="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EXP/../.." && pwd)"                                   # concept_dim
FORK="$(cd "$ROOT/${FORK:-../geometry-of-refusal}" && pwd)"
SRC="$(cd "$ROOT/${SRC:-../shared-refusal-component}" && pwd)"
OUT="${OUT:-$FORK/results/loss_terms}"
export FORK                                                         # shared_refusal.artifacts.FORK

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_CACHE_DIR="${HUGGINGFACE_CACHE_DIR:-$HF_HOME/hub}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$OUT/logs" "$EXP/reports"

# py <script> <args…> — запуск скрипта эксперимента в окружении shared-refusal-component.
py() { (cd "$SRC" && uv run python "$EXP/scripts/$1" "${@:2}"); }

# need_gpu — GPU обязателен и задаёт единственную видимую карту.
need_gpu() {
  : "${GPU:?укажи GPU=<индекс> (эксперимент идёт на одной карте)}"
  export CUDA_VISIBLE_DEVICES="$GPU"
}
