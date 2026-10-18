#!/usr/bin/env bash
set -euo pipefail

script_source="${BASH_SOURCE[0]}"
script_parent="."
if [[ "${script_source}" == */* ]]; then
  script_parent="${script_source%/*}"
fi
MATRIX_SCRIPT_DIR="$(cd -- "${script_parent}" && pwd)"

MATRIX_GAME_NAME="freeway"
MATRIX_GAME_LABEL="Freeway"
MATRIX_DEFAULT_TIMESTEPS=20000000
MATRIX_DEFAULT_GROUP="freeway_accel_matrix_20m_20261018"
MATRIX_DEFAULT_SAVE_ROOT="./models/freeway_accel_matrix_20m"
MATRIX_BASE_CONFIG="pqn_jaxatari_object_freeway_eval"
MATRIX_ACCEL_CONFIG="pqn_jaxatari_object_jaxued_accel_freeway"
MATRIX_EVAL_SEED=20261014
MATRIX_EVAL_MODS=(stop_all_cars invert_speed static_cars)
MATRIX_PROFILES=(size9 speed3 full27)

source "${MATRIX_SCRIPT_DIR}/accel_matrix_common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/benchmarks/run_freeway_accel_matrix.sh [OPTIONS]

Run one plain Freeway baseline and three ACCEL curricula:
  size9    car width x height (3 x 3), native speed
  speed3   three car-speed multipliers, native geometry
  full27   speed x width x height (3 x 3 x 3)

Every run evaluates base, stop_all_cars, invert_speed, and static_cars.
EOF
  matrix_common_options
}

matrix_initialize "$@"
matrix_print_plan

if [[ "${MATRIX_RUN_BASELINE}" == true ]]; then
  matrix_run_baseline
fi

for profile in "${MATRIX_PROFILES[@]}"; do
  if ! matrix_profile_selected "${profile}"; then
    continue
  fi
  case "${profile}" in
    size9)
      matrix_run_accel "${profile}" 9 0.7 0.6 0.05 \
        'ACCEL_PARAMETER_CAR_SPEED_MULTIPLIERS=[1.0]' \
        'ACCEL_PARAMETER_CAR_WIDTHS=[8,10,12]' \
        'ACCEL_PARAMETER_CAR_HEIGHTS=[10,12,14]'
      ;;
    speed3)
      matrix_run_accel "${profile}" 3 0.7 0.6 0.05 \
        'ACCEL_PARAMETER_CAR_SPEED_MULTIPLIERS=[1.0,1.5,2.0]' \
        'ACCEL_PARAMETER_CAR_WIDTHS=[8]' \
        'ACCEL_PARAMETER_CAR_HEIGHTS=[10]'
      ;;
    full27)
      matrix_run_accel "${profile}" 27 0.7 0.6 0.05 \
        'ACCEL_PARAMETER_CAR_SPEED_MULTIPLIERS=[1.0,1.5,2.0]' \
        'ACCEL_PARAMETER_CAR_WIDTHS=[8,10,12]' \
        'ACCEL_PARAMETER_CAR_HEIGHTS=[10,12,14]'
      ;;
  esac
done

matrix_finish
