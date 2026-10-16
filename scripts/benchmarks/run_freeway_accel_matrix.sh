#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/benchmarks/run_freeway_accel_matrix.sh [OPTIONS]

Train the strongest object-centric Freeway ACCEL curricula together with an equally
budgeted plain PQN baseline. Every run is evaluated on normal Freeway,
`slow_cars`, and `stop_all_cars` using the same held-out evaluation seed.

ACCEL profiles:
  size9     best geometry grid:                    3 * 3 = 9 levels
  speed3    second-best speed grid:                        3 levels
  full27    legacy speed * width * height grid: 3 * 3 * 3 = 27 levels
  winners   run size9 and speed3 (default)
  all       run size9, speed3, and legacy full27

The size9 and speed3 level spaces are kept unchanged so the 20M results remain
comparable to the successful runs. Individual failures are recorded while the
remaining selected runs continue. Models/configs are saved beneath separate
per-run directories to prevent checkpoint overwrites.

Options:
  --profile PROFILE    one profile or suite above (default: winners)
  --skip-baseline      omit the plain Freeway baseline
  --baseline-only      run only the plain Freeway baseline
  --timesteps N        environment steps per training (default: 20000000)
  --group NAME         W&B group (default: freeway_winners_20m_20261016)
  --save-root PATH     checkpoint root (default: ./models/freeway_winners_20m)
  --smoke              short validation run; not a reportable experiment
  -h, --help           show this help
EOF
}

selected_profile="winners"
run_baseline=true
run_accel=true
smoke=false
total_timesteps=20000000
wandb_group="freeway_winners_20m_20261016"
save_root="./models/freeway_winners_20m"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ $# -lt 2 ]]; then
        echo "Error: --profile requires size9, speed3, full27, winners, or all." >&2
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
    --timesteps)
      if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --timesteps requires a positive integer." >&2
        exit 2
      fi
      total_timesteps="$2"
      shift 2
      ;;
    --group)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Error: --group requires a non-empty W&B group name." >&2
        exit 2
      fi
      wandb_group="$2"
      shift 2
      ;;
    --save-root)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Error: --save-root requires a non-empty path." >&2
        exit 2
      fi
      save_root="$2"
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
  size9|speed3|full27|winners|all) ;;
  *)
    echo "Error: invalid --profile '${selected_profile}'." >&2
    exit 2
    ;;
esac

if [[ "${run_baseline}" == false && "${run_accel}" == false ]]; then
  echo "Error: --skip-baseline and --baseline-only disable every experiment." >&2
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
  echo "Error: reportable experiments require CUDA (JAX_PLATFORMS=${jax_platforms})." >&2
  exit 2
fi
if [[ ",${jax_platforms}," != *,cpu,* ]]; then
  jax_platforms="${jax_platforms},cpu"
fi
export JAX_PLATFORMS="${jax_platforms}"
export PYTHONPATH="${script_dir}:${repo_root}/src:${repo_root}/jaxued/jaxued/src${PYTHONPATH:+:${PYTHONPATH}}"

# Avoid selecting a forward-compatibility libcuda from LD_LIBRARY_PATH. This
# caused CUDA error 804 on older pool GPUs in the Seaquest/Pong experiments.
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
    printf '%s\n' \
      "CUDA preflight failed before ${run_label}." \
      'The experiment was skipped rather than silently running on CPU.' >&2
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
    echo "FAILED: ${run_label}. Continuing with the next selected training." >&2
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

pre_baseline_profiles=()
post_baseline_profiles=()
case "${selected_profile}" in
  winners)
    pre_baseline_profiles=(size9 speed3)
    ;;
  all)
    pre_baseline_profiles=(size9 speed3)
    post_baseline_profiles=(full27)
    ;;
  *)
    pre_baseline_profiles=("${selected_profile}")
    ;;
esac
profile_names=("${pre_baseline_profiles[@]}" "${post_baseline_profiles[@]}")

if ((total_timesteps % 1000000 == 0)); then
  budget_label="$((total_timesteps / 1000000))m"
else
  budget_label="${total_timesteps}steps"
fi

planned_runs=0
if [[ "${run_accel}" == true ]]; then
  planned_runs=${#profile_names[@]}
fi
if [[ "${run_baseline}" == true ]]; then
  planned_runs=$((planned_runs + 1))
fi
planned_env_steps=$((planned_runs * total_timesteps))

echo "Freeway object-centric ACCEL comparison matrix"
echo "  ACCEL profile:    ${selected_profile}"
echo "  W&B group:        ${wandb_group}"
echo "  checkpoint root:  ${save_root}"
echo "  JAX platforms:    ${JAX_PLATFORMS}"
echo "  steps per run:    ${total_timesteps}"
echo "  selected runs:    ${planned_runs}"
echo "  planned steps:    ${planned_env_steps}"
echo "  evaluation suite: base, slow_cars, stop_all_cars"
echo "  baseline:         ${run_baseline}"
echo "  ACCEL variants:   ${run_accel}"
echo "  smoke only:       ${smoke}"

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

run_accel_profile() {
  local profile_name="$1"
  local speed_values
  local width_values
  local height_values
  local buffer_capacity

  case "${profile_name}" in
    full27)
      speed_values='[1.0,1.5,2.0]'
      width_values='[8,10,12]'
      height_values='[10,12,14]'
      buffer_capacity=27
      ;;
    speed3)
      speed_values='[1.0,1.5,2.0]'
      width_values='[8]'
      height_values='[10]'
      buffer_capacity=3
      ;;
    size9)
      speed_values='[1.0]'
      width_values='[8,10,12]'
      height_values='[10,12,14]'
      buffer_capacity=9
      ;;
  esac

  local run_label="freeway_object_accel_${profile_name}_${budget_label}"
  run_experiment "${run_label}" \
    "${script_dir}/jaxatari_accel_plr.py" \
    --config-name=pqn_jaxatari_object_jaxued_accel_freeway \
    "NAME=${run_label}" \
    "+PROFILE=${profile_name}" \
    "TOTAL_TIMESTEPS=${total_timesteps}" \
    "TOTAL_TIMESTEPS_DECAY=${total_timesteps}" \
    "ACCEL_PARAMETER_CAR_SPEED_MULTIPLIERS=${speed_values}" \
    "ACCEL_PARAMETER_CAR_WIDTHS=${width_values}" \
    "ACCEL_PARAMETER_CAR_HEIGHTS=${height_values}" \
    "ACCEL_BUFFER_CAPACITY=${buffer_capacity}" \
    "SAVE_PATH=${save_root}/${profile_name}" \
    "+WANDB_GROUP=${wandb_group}" \
    "+WANDB_JOB_TYPE=accel_${profile_name}" \
    "${accel_smoke_overrides[@]}"
}

# Run the strongest profiles before the baseline so they complete first if an
# allocation ends early.
if [[ "${run_accel}" == true ]]; then
  for profile_name in "${pre_baseline_profiles[@]}"; do
    run_accel_profile "${profile_name}"
  done
fi

if [[ "${run_baseline}" == true ]]; then
  baseline_label="freeway_object_base_eval_${budget_label}"
  run_experiment "${baseline_label}" \
    "${script_dir}/pqn_agent.py" \
    +alg=pqn_jaxatari_object_freeway_eval \
    "alg.NAME=${baseline_label}" \
    "alg.TOTAL_TIMESTEPS=${total_timesteps}" \
    "alg.TOTAL_TIMESTEPS_DECAY=${total_timesteps}" \
    NUM_SEEDS=1 \
    SEED=42 \
    ENTITY=null \
    PROJECT=Thesis \
    WANDB_MODE=online \
    "SAVE_PATH=${save_root}/baseline" \
    "+WANDB_GROUP=${wandb_group}" \
    +WANDB_JOB_TYPE=baseline \
    "${baseline_smoke_overrides[@]}"
fi

# The optional legacy full27 profile runs only after the directly comparable
# baseline has completed.
if [[ "${run_accel}" == true ]]; then
  for profile_name in "${post_baseline_profiles[@]}"; do
    run_accel_profile "${profile_name}"
  done
fi

echo
echo "Freeway object-centric ACCEL matrix finished."
echo "Completed runs (${#completed_runs[@]}):"
if ((${#completed_runs[@]} > 0)); then
  printf '  - %s\n' "${completed_runs[@]}"
else
  echo "  (none)"
fi

echo "Failed runs (${#failed_runs[@]}):"
if ((${#failed_runs[@]} > 0)); then
  printf '  - %s\n' "${failed_runs[@]}"
  echo "All selected runs were attempted; inspect the failures above." >&2
  exit 1
else
  echo "  (none)"
fi
