#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/benchmarks/run_seaquest_pong_port_matrix.sh [OPTIONS]

Run the object-centric Pong ACCEL reference first, followed by the Seaquest
baseline and three focused 32-level ACCEL retests. Pixel jobs are intentionally
excluded because their failure on mlsp7 is a Pascal/cuDNN compatibility issue,
not an ACCEL result.

Core profiles (included by `all`):
  mild                 exact best-known mild Seaquest profile
  matched              mild curriculum with base/Pong protagonist settings
  matched_gentle       matched protagonist with the successful gentle grid

Additional matched ablation:
  matched_mutation45   matched, except mutation probability 0.55 -> 0.45

Additional one-factor mild follow-ups:
  mild_lr_decay        mild, except enable linear learning-rate decay
  mild_layer_norm      mild, except use layer norm instead of batch norm
  mild_gentle_physics  mild, except use a gentler 32-level physics grid

Pong reference:
  pong_wandb_reference exact supplied 15M parameter-vector training config

The Seaquest runs share base/gravity evaluations, a 32-level physics grid, and
a 20M environment-step budget. The Pong reference retains its supplied 15M
configuration unchanged. Every run uses seed 42 and the same W&B group. If one
run fails, the launcher records it and continues with the next run.

Options:
  --profile PROFILE  one profile above, or all for the core three (default: all)
  --skip-baseline    omit the plain Seaquest baseline
  --baseline-only    run only the evaluated plain PQN baseline
  --skip-pong        omit the Pong reference training
  --pong-only        run only the Pong reference training
  --group NAME       W&B group (default: seaquest_pong_first_20m_20261014)
  --smoke            short validation run; not a reportable experiment
  -h, --help         show this help
EOF
}

selected_profile="all"
run_baseline=true
run_accel=true
run_pong=true
smoke=false
pong_total_timesteps=15000000
seaquest_total_timesteps=20000000
wandb_group="seaquest_pong_first_20m_20261014"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ $# -lt 2 ]]; then
        echo "Error: --profile requires a listed profile or all." >&2
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
      run_pong=false
      shift
      ;;
    --skip-pong)
      run_pong=false
      shift
      ;;
    --pong-only)
      run_baseline=false
      run_accel=false
      run_pong=true
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
  mild|matched|matched_gentle|matched_mutation45|mild_lr_decay|mild_layer_norm|mild_gentle_physics|all) ;;
  *)
    echo "Error: invalid --profile '${selected_profile}'." >&2
    exit 2
    ;;
esac

if [[ "${run_baseline}" == false && "${run_accel}" == false && "${run_pong}" == false ]]; then
  echo "Error: the selected options disable every experiment." >&2
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

# The pip CUDA plugin must not pick up a forward-compatibility libcuda through
# LD_LIBRARY_PATH (CUDA error 804 on unsupported GPUs).
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
      'CUDA preflight failed. Do not run this experiment on CPU.' \
      'Run `nvidia-smi`; if it also fails, use a healthy pool machine or contact' \
      'the administrator. If it succeeds, test JAX with:' \
      '' \
      '  env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda python -c' \
      "    'import jax; print(jax.devices())'" >&2
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
    echo "FAILED: ${run_label} exited with status ${exit_status}. Continuing with the next selected training." >&2
  fi

  # A failed individual experiment must not trigger the launcher's `set -e`.
  return 0
}

echo "Pong reference first, followed by Seaquest object-centric retests"
echo "  ACCEL profile:  ${selected_profile}"
echo "  W&B group:      ${wandb_group}"
echo "  JAX platforms:  ${JAX_PLATFORMS}"
echo "  Pong steps:     ${pong_total_timesteps}"
echo "  Seaquest steps: ${seaquest_total_timesteps}"
echo "  baseline:       ${run_baseline}"
echo "  ACCEL variants: ${run_accel}"
echo "  Pong reference: ${run_pong}"
echo "  smoke only:     ${smoke}"

baseline_smoke_overrides=()
accel_smoke_overrides=()
pong_smoke_overrides=()
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
  pong_smoke_overrides=(
    TOTAL_TIMESTEPS=262144
    TOTAL_TIMESTEPS_DECAY=262144
    EVAL_DURING_TRAINING=false
    FINAL_EVAL=true
    +EVAL_NUM_ENVS=16
    +EVAL_NUM_STEPS=1000
    EVAL_RECORD_VIDEO=false
    TRAIN_RECORD_ACCEL_VIDEO=false
  )
fi

# Reproduce the supplied W&B Pong run. The compatibility overrides below
# neutralize curriculum/gameplay dimensions added after that recorded run:
# lazy tracking remains enabled because [0,1] was part of the supplied grid;
# enemy cadence is omitted, match-ball-speed stays off, and randomized vertical
# serves are enabled explicitly for this new training run.
pong_reference_args=(
  "TOTAL_TIMESTEPS=${pong_total_timesteps}"
  "TOTAL_TIMESTEPS_DECAY=${pong_total_timesteps}"
  NUM_ENVS=128
  NUM_STEPS=128
  EPS_START=1.0
  EPS_FINISH=0.02
  EPS_DECAY=0.5
  NUM_EPOCHS=5
  NUM_MINIBATCHES=32
  NORM_TYPE=layer_norm
  LR=0.000225
  LR_LINEAR_DECAY=true
  MAX_GRAD_NORM=10
  GAMMA=0.99
  LAMBDA=0.65
  ENV_NAME=pong
  OBJECT_CENTRIC=true
  JAXUED_ALG=accel
  ACCEL_LEVEL_ENCODING=parameter_vector
  ACCEL_BUFFER_CAPACITY=1024
  ACCEL_MUTATION_MULTI_PARAM_PROB=0.05
  ACCEL_MUTATION_PROB=0.6
  ACCEL_PARAMETER_BASELINE_PROB=0.7
  'ACCEL_PARAMETER_DELTA_MULTIPLIERS=[1,1.05,1.1,1.15,1.2,1.25]'
  'ACCEL_PARAMETER_ENEMY_STEP_SIZES=[1,2,3,4]'
  'ACCEL_PARAMETER_ENEMY_TRACKING_MODES=[0,1]'
  'ACCEL_PARAMETER_PADDLE_HEIGHTS=[14,15,16]'
  'ACCEL_PARAMETER_PADDLE_WIDTHS=[2,3,4]'
  ACCEL_ENABLE_LAZY_ENEMY=true
  +ACCEL_ENABLE_ENEMY_MOVE_INTERVAL=false
  'ACCEL_PARAMETER_ENEMY_MOVE_INTERVALS=[0]'
  ACCEL_ENEMY_MATCH_BALL_SPEED=false
  ACCEL_RANDOMIZE_SERVE_DIRECTION=true
  REPLAY_PROB=0.7
  STALENESS_COEFF=0.35
  MINIMUM_FILL_RATIO=0.35
  TEMPERATURE=1.0
  JAXUED_SCORE_FUNCTION=q_pvl
  EXPLORATORY_GRAD_UPDATES=true
  EVAL_DURING_TRAINING=true
  EVAL_EVERY_STEPS=1000000
  FINAL_EVAL=true
  EVAL_RECORD_VIDEO=true
  TRAIN_RECORD_ACCEL_VIDEO=true
  +HYP_TUNE=false
  +ENTITY=null
  +PROJECT=Thesis
  +WANDB_MODE=online
  +SEED=42
  +NUM_SEEDS=1
)

core_profile_names=(mild matched matched_gentle)
if [[ "${selected_profile}" == all ]]; then
  profile_names=("${core_profile_names[@]}")
else
  profile_names=("${selected_profile}")
fi
profile_args=()

build_profile_args() {
  local profile_name="$1"
  local num_envs
  local num_steps
  local eps_start
  local eps_finish
  local eps_decay
  local num_epochs
  local num_minibatches
  local norm_type
  local learning_rate
  local lr_linear_decay
  local max_grad_norm
  local gamma
  local lambda
  local baseline_prob=0.65
  local mutation_prob
  local mutation_multi_param_prob=0.08
  local replay_prob=0.65
  local staleness_coeff=0.30
  local minimum_fill_ratio=0.30
  local temperature=0.90
  local player_move_speeds='[1,3]'
  local enemy_speed_multipliers='[1,3]'
  local player_missile_speeds='[2,4,5,7]'
  local enemy_missile_speed_multipliers='[1,3]'

  case "${profile_name}" in
    mild|mild_lr_decay|mild_layer_norm|mild_gentle_physics)
      num_envs=120
      num_steps=120
      eps_start=0.98
      eps_finish=0.025
      eps_decay=0.45
      num_epochs=4
      num_minibatches=30
      norm_type=batch_norm
      learning_rate=0.000200
      lr_linear_decay=false
      max_grad_norm=9
      gamma=0.985
      lambda=0.60
      mutation_prob=0.55
      case "${profile_name}" in
        mild_lr_decay)
          lr_linear_decay=true
          ;;
        mild_layer_norm)
          norm_type=layer_norm
          ;;
        mild_gentle_physics)
          # Keep 2*2*4*2 = 32 levels, but avoid the harsh 3x velocity bins.
          player_move_speeds='[1,2]'
          enemy_speed_multipliers='[1,2]'
          player_missile_speeds='[3,4,5,6]'
          enemy_missile_speed_multipliers='[1,2]'
          ;;
      esac
      ;;
    matched|matched_gentle)
      # Exactly match the plain baseline/Pong protagonist. The curriculum is
      # then the only systematic training difference from the base run.
      num_envs=128
      num_steps=128
      eps_start=1.0
      eps_finish=0.02
      eps_decay=0.5
      num_epochs=5
      num_minibatches=32
      norm_type=layer_norm
      learning_rate=0.000225
      lr_linear_decay=true
      max_grad_norm=10
      gamma=0.99
      lambda=0.65
      mutation_prob=0.55
      if [[ "${profile_name}" == matched_gentle ]]; then
        # Reproduce the strongest completed ACCEL candidate: matched training
        # settings, a gentler 32-level grid, and its original sampler values.
        player_move_speeds='[1,2]'
        enemy_speed_multipliers='[1,2]'
        player_missile_speeds='[3,4,5,6]'
        enemy_missile_speed_multipliers='[1,2]'
        baseline_prob=0.70
        mutation_prob=0.60
        mutation_multi_param_prob=0.05
        replay_prob=0.70
        staleness_coeff=0.35
        minimum_fill_ratio=0.35
        temperature=1.0
      fi
      ;;
    matched_mutation45)
      # One-factor ablation from `matched`: slightly longer replay refinement
      # before a replayed batch is mutated.
      num_envs=128
      num_steps=128
      eps_start=1.0
      eps_finish=0.02
      eps_decay=0.5
      num_epochs=5
      num_minibatches=32
      norm_type=layer_norm
      learning_rate=0.000225
      lr_linear_decay=true
      max_grad_norm=10
      gamma=0.99
      lambda=0.65
      mutation_prob=0.45
      ;;
  esac

  profile_args=(
    "TOTAL_TIMESTEPS=${seaquest_total_timesteps}"
    "TOTAL_TIMESTEPS_DECAY=${seaquest_total_timesteps}"
    "NUM_ENVS=${num_envs}"
    "NUM_STEPS=${num_steps}"
    "EPS_START=${eps_start}"
    "EPS_FINISH=${eps_finish}"
    "EPS_DECAY=${eps_decay}"
    "NUM_EPOCHS=${num_epochs}"
    "NUM_MINIBATCHES=${num_minibatches}"
    "NORM_TYPE=${norm_type}"
    "LR=${learning_rate}"
    "LR_LINEAR_DECAY=${lr_linear_decay}"
    "MAX_GRAD_NORM=${max_grad_norm}"
    "GAMMA=${gamma}"
    "LAMBDA=${lambda}"
    ACCEL_BUFFER_CAPACITY=32
    "ACCEL_PARAMETER_BASELINE_PROB=${baseline_prob}"
    "ACCEL_MUTATION_PROB=${mutation_prob}"
    "ACCEL_MUTATION_MULTI_PARAM_PROB=${mutation_multi_param_prob}"
    "REPLAY_PROB=${replay_prob}"
    "STALENESS_COEFF=${staleness_coeff}"
    "MINIMUM_FILL_RATIO=${minimum_fill_ratio}"
    "TEMPERATURE=${temperature}"
    "ACCEL_PARAMETER_PLAYER_MOVE_SPEEDS=${player_move_speeds}"
    "ACCEL_PARAMETER_ENEMY_SPEED_MULTIPLIERS=${enemy_speed_multipliers}"
    "ACCEL_PARAMETER_PLAYER_MISSILE_SPEEDS=${player_missile_speeds}"
    "ACCEL_PARAMETER_ENEMY_MISSILE_SPEED_MULTIPLIERS=${enemy_missile_speed_multipliers}"
  )
}

if [[ "${run_pong}" == true ]]; then
  run_experiment "pong_object_accel_wandb_reference" \
    "${script_dir}/jaxatari_accel_plr.py" \
    --config-name=pqn_jaxatari_object_jaxued_accel \
    +NAME=pong_object_accel_wandb_reference \
    "+WANDB_GROUP=${wandb_group}" \
    +WANDB_JOB_TYPE=pong_reference \
    "${pong_reference_args[@]}" \
    "${pong_smoke_overrides[@]}"
fi

if [[ "${run_baseline}" == true ]]; then
  run_experiment "seaquest_object_base_eval_retest" \
    "${script_dir}/pqn_agent.py" \
    +alg=pqn_jaxatari_object_seaquest_eval \
    alg.NAME=seaquest_object_base_eval_retest \
    "alg.TOTAL_TIMESTEPS=${seaquest_total_timesteps}" \
    "alg.TOTAL_TIMESTEPS_DECAY=${seaquest_total_timesteps}" \
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

if [[ "${run_accel}" == true ]]; then
  for profile_name in "${profile_names[@]}"; do
    if [[ "${selected_profile}" != "all" && "${selected_profile}" != "${profile_name}" ]]; then
      continue
    fi
    build_profile_args "${profile_name}"
    run_experiment "seaquest_object_accel32_${profile_name}_retest" \
      "${script_dir}/jaxatari_accel_plr.py" \
      --config-name=pqn_jaxatari_object_jaxued_accel_seaquest \
      "+NAME=seaquest_object_accel32_${profile_name}_retest" \
      "PROFILE=${profile_name}" \
      "+WANDB_GROUP=${wandb_group}" \
      "+WANDB_JOB_TYPE=accel_${profile_name}" \
      "${profile_args[@]}" \
      "${accel_smoke_overrides[@]}"
  done
fi

echo
echo "Pong/Seaquest object-centric matrix finished."
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
