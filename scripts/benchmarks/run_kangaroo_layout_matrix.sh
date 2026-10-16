#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/benchmarks/run_kangaroo_layout_matrix.sh [OPTIONS]

Run one plain object-centric Kangaroo baseline followed by three structured
ACCEL trainings. The ACCEL runs differ only in horizontal object edit size:
  object_step4     4-pixel fruit/bell edits
  object_step8     8-pixel fruit/bell edits (framework default)
  object_step12   12-pixel fruit/bell edits

Every run evaluates both normal Kangaroo and the `second_level_only` mod.
Individual failures are recorded while the remaining selected runs continue.

Options:
  --profile PROFILE  one profile above, or all (default: all)
  --skip-baseline    run only the selected ACCEL training(s)
  --baseline-only    run only the plain evaluated baseline
  --group NAME       W&B group (default: kangaroo_layout_matrix_20261013)
  --smoke            short validation run; not a reportable experiment
  -h, --help         show this help
EOF
}

selected_profile="all"
run_baseline=true
run_accel=true
smoke=false
wandb_group="kangaroo_layout_matrix_20261013"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ $# -lt 2 ]]; then
        echo "Error: --profile requires object_step4, object_step8, object_step12, or all." >&2
        exit 2
      fi
      selected_profile="$2"
      shift 2
      ;;
    --skip-baseline)
      run_baseline=false
      shift
      ;;
    --baseline-only)
      run_accel=false
      shift
      ;;
    --group)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Error: --group requires a non-empty W&B group name." >&2
        exit 2
      fi
      wandb_group="$2"
      shift 2
      ;;
    --smoke)
      smoke=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'." >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${selected_profile}" in
  object_step4|object_step8|object_step12|all) ;;
  *)
    echo "Error: invalid --profile '${selected_profile}'." >&2
    exit 2
    ;;
esac

if [[ "${run_baseline}" == false && "${run_accel}" == false ]]; then
  echo "Error: --skip-baseline and --baseline-only cannot be combined." >&2
  exit 2
fi

script_source="${BASH_SOURCE[0]}"
script_parent="."
if [[ "${script_source}" == */* ]]; then
  script_parent="${script_source%/*}"
fi
script_dir="$(cd -- "${script_parent}" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

cd "${repo_root}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Error: Python executable '${python_bin}' was not found." >&2
  echo "Activate ~/JAXAtariAdversarialThesis/.venv or set PYTHON_BIN." >&2
  exit 127
fi

jax_platforms="${JAX_PLATFORMS:-cuda,cpu}"
if [[ ",${jax_platforms}," != *,cuda,* ]]; then
  echo "Error: these reportable experiments require CUDA (JAX_PLATFORMS=${jax_platforms})." >&2
  exit 2
fi
if [[ ",${jax_platforms}," != *,cpu,* ]]; then
  jax_platforms="${jax_platforms},cpu"
fi
export JAX_PLATFORMS="${jax_platforms}"
export PYTHONPATH="${script_dir}:${repo_root}/src:${repo_root}/jaxued/jaxued/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  echo "Ignoring LD_LIBRARY_PATH for JAX GPU processes: ${LD_LIBRARY_PATH}"
fi

gpu_preflight() {
  local run_label="$1"
  echo "Checking CUDA before ${run_label}..."
  if ! (
    unset LD_LIBRARY_PATH
    export JAX_PLATFORMS=cuda
    "${python_bin}" -c \
      'import jax; devices = jax.devices(); assert devices and all(d.platform in ("cuda", "gpu") for d in devices), devices; print("CUDA preflight OK:", devices)'
  ); then
    echo "CUDA preflight failed for ${run_label}; the run will be skipped." >&2
    return 1
  fi
}

run_gpu_python() {
  (
    unset LD_LIBRARY_PATH
    "${python_bin}" "$@"
  )
}

completed_runs=()
failed_runs=()

run_experiment() {
  local run_label="$1"
  shift

  echo
  echo "Starting ${run_label}..."
  if ! gpu_preflight "${run_label}"; then
    failed_runs+=("${run_label} (CUDA preflight failed)")
    return 0
  fi

  local exit_status=0
  if run_gpu_python "$@"; then
    completed_runs+=("${run_label}")
    echo "Completed: ${run_label}."
  else
    exit_status=$?
    failed_runs+=("${run_label} (exit ${exit_status})")
    echo "FAILED: ${run_label} exited with status ${exit_status}. Continuing." >&2
  fi
  return 0
}

baseline_smoke_overrides=()
accel_smoke_overrides=()
if [[ "${smoke}" == true ]]; then
  baseline_smoke_overrides=(
    alg.TOTAL_TIMESTEPS=262144
    alg.TOTAL_TIMESTEPS_DECAY=262144
    alg.EVAL_DURING_TRAINING=false
    alg.FINAL_EVAL=true
    alg.EVAL_NUM_ENVS=16
    alg.EVAL_NUM_STEPS=1000
    alg.EVAL_RECORD_VIDEO=false
    alg.RECORD_FINAL_VIDEO=false
  )
  accel_smoke_overrides=(
    TOTAL_TIMESTEPS=262144
    TOTAL_TIMESTEPS_DECAY=262144
    EVAL_DURING_TRAINING=false
    FINAL_EVAL=true
    EVAL_NUM_ENVS=16
    EVAL_NUM_STEPS=1000
    EVAL_RECORD_VIDEO=false
    TRAIN_RECORD_ACCEL_VIDEO=false
  )
fi

echo "Kangaroo structured-layout matrix"
echo "  ACCEL profile:  ${selected_profile}"
echo "  W&B group:      ${wandb_group}"
echo "  JAX platforms:  ${JAX_PLATFORMS}"
echo "  baseline:       ${run_baseline}"
echo "  ACCEL variants: ${run_accel}"
echo "  smoke only:     ${smoke}"

if [[ "${run_baseline}" == true ]]; then
  run_experiment "kangaroo_object_base" \
    "${script_dir}/pqn_agent.py" \
    +alg=pqn_jaxatari_object_kangaroo_eval \
    alg.NAME=kangaroo_object_base \
    NUM_SEEDS=1 \
    SEED=42 \
    ENTITY=null \
    PROJECT=Thesis \
    WANDB_MODE=online \
    SAVE_PATH=null \
    "+WANDB_GROUP=${wandb_group}" \
    +WANDB_JOB_TYPE=baseline \
    "${baseline_smoke_overrides[@]}"
fi

profile_names=(object_step4 object_step8 object_step12)
object_steps=(4 8 12)

if [[ "${run_accel}" == true ]]; then
  for index in "${!profile_names[@]}"; do
    profile_name="${profile_names[$index]}"
    if [[ "${selected_profile}" != "all" && "${selected_profile}" != "${profile_name}" ]]; then
      continue
    fi
    object_step="${object_steps[$index]}"
    run_experiment "kangaroo_object_accel_${profile_name}" \
      "${script_dir}/jaxatari_accel_plr.py" \
      --config-name=pqn_jaxatari_object_jaxued_accel_kangaroo \
      "+NAME=kangaroo_object_accel_${profile_name}" \
      "PROFILE=${profile_name}" \
      "ACCEL_KANGAROO_OBJECT_STEP=${object_step}" \
      "+WANDB_GROUP=${wandb_group}" \
      "+WANDB_JOB_TYPE=accel_${profile_name}" \
      "${accel_smoke_overrides[@]}"
  done
fi

echo
echo "Kangaroo structured-layout matrix finished."
echo "Completed runs (${#completed_runs[@]}):"
if ((${#completed_runs[@]} > 0)); then
  printf '  - %s\n' "${completed_runs[@]}"
else
  echo "  (none)"
fi

echo "Failed runs (${#failed_runs[@]}):"
if ((${#failed_runs[@]} > 0)); then
  printf '  - %s\n' "${failed_runs[@]}"
  exit 1
else
  echo "  (none)"
fi
