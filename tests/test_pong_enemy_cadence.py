import jax
import jax.numpy as jnp

from jaxatari.accel_wrappers import PongAccelModWrapper
from jaxatari.environment import JaxEnvironment
from jaxatari.games.jax_pong import JaxPong, PongConstants
from jaxatari.wrappers import AtariWrapper, _get_core_env_state, _set_core_env_state


def _core_without_renderer():
    """Construct Pong logic without requiring the optional sprite bundle."""
    env = object.__new__(JaxPong)
    JaxEnvironment.__init__(env, PongConstants())
    return env


def test_enemy_move_interval_uses_native_or_every_nth_schedule():
    env = JaxPong()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        enemy_y=jnp.asarray(100, dtype=jnp.int32),
        ball_y=jnp.asarray(120, dtype=jnp.int32),
        enemy_step_size=jnp.asarray(2, dtype=jnp.int32),
    )

    native_move = env._enemy_step(
        state.replace(
            step_counter=jnp.asarray(1, dtype=jnp.int32),
            enemy_move_interval=jnp.asarray(0, dtype=jnp.int32),
        )
    )
    native_skip = env._enemy_step(
        state.replace(
            step_counter=jnp.asarray(8, dtype=jnp.int32),
            enemy_move_interval=jnp.asarray(0, dtype=jnp.int32),
        )
    )
    every_fourth_move = env._enemy_step(
        state.replace(
            step_counter=jnp.asarray(4, dtype=jnp.int32),
            enemy_move_interval=jnp.asarray(4, dtype=jnp.int32),
        )
    )
    every_fourth_skip = env._enemy_step(
        state.replace(
            step_counter=jnp.asarray(5, dtype=jnp.int32),
            enemy_move_interval=jnp.asarray(4, dtype=jnp.int32),
        )
    )

    assert int(native_move.enemy_y) == 102
    assert int(native_skip.enemy_y) == 100
    assert int(every_fourth_move.enemy_y) == 102
    assert int(every_fourth_skip.enemy_y) == 100


def test_fast_incoming_ball_safety_overrides_a_slow_enemy_cadence():
    env = JaxPong(PongConstants(ENEMY_MATCH_BALL_SPEED=True))
    _, state = env.reset(jax.random.PRNGKey(1))
    state = state.replace(
        step_counter=jnp.asarray(1, dtype=jnp.int32),
        enemy_y=jnp.asarray(24, dtype=jnp.int32),
        enemy_move_interval=jnp.asarray(4, dtype=jnp.int32),
        enemy_tracking_mode=jnp.asarray(1, dtype=jnp.int32),
        ball_x=jnp.asarray(20, dtype=jnp.int32),
        ball_y=jnp.asarray(150, dtype=jnp.int32),
        ball_vel_x=jnp.asarray(-4.0, dtype=jnp.float32),
        ball_vel_y=jnp.asarray(3.0, dtype=jnp.float32),
    )

    without_safety = JaxPong()._enemy_step(state)
    enemy_ready = env._enemy_step(state)
    returned_ball = env._ball_step(enemy_ready, jnp.asarray(0, dtype=jnp.int32))

    assert int(without_safety.enemy_y) == 24
    assert int(enemy_ready.enemy_y) == 145
    assert float(returned_ball.ball_vel_x) == 4.0


def test_fast_incoming_ball_visibly_increases_enemy_tracking_speed():
    env = JaxPong(PongConstants(ENEMY_MATCH_BALL_SPEED=True))
    _, state = env.reset(jax.random.PRNGKey(5))
    state = state.replace(
        step_counter=jnp.asarray(1, dtype=jnp.int32),
        enemy_y=jnp.asarray(100, dtype=jnp.int32),
        enemy_step_size=jnp.asarray(2, dtype=jnp.int32),
        # A slow cadence and lazy tracking must not suppress the fast-ball
        # response.  The ball still has enough horizontal travel that the
        # paddle should move at its speed-matched 8 px/frame, not teleport.
        enemy_move_interval=jnp.asarray(4, dtype=jnp.int32),
        enemy_tracking_mode=jnp.asarray(1, dtype=jnp.int32),
        ball_x=jnp.asarray(75, dtype=jnp.int32),
        ball_y=jnp.asarray(170, dtype=jnp.int32),
        ball_vel_x=jnp.asarray(-4.0, dtype=jnp.float32),
        ball_vel_y=jnp.asarray(0.0, dtype=jnp.float32),
    )

    enemy_after = env._enemy_step(state)

    # 2 px/frame base tracking scales with a 4 px/frame incoming ball to
    # 8 px/frame, making the catch-up movement observable in training videos.
    assert int(enemy_after.enemy_y) == 108
    assert int(enemy_after.enemy_speed) == 8


def test_match_ball_speed_bypasses_slow_enemy_before_rounding_accelerates_ball():
    env = JaxPong(PongConstants(ENEMY_MATCH_BALL_SPEED=True))
    _, state = env.reset(jax.random.PRNGKey(6))
    state = state.replace(
        step_counter=jnp.asarray(1, dtype=jnp.int32),
        enemy_y=jnp.asarray(100, dtype=jnp.int32),
        # Both level settings slow normal enemy movement. The incoming ball
        # has raw base speed, modelling a small ACCEL multiplier that rounds
        # to one pixel rather than increasing the physical position delta.
        enemy_step_size=jnp.asarray(1, dtype=jnp.int32),
        enemy_move_interval=jnp.asarray(3, dtype=jnp.int32),
        enemy_tracking_mode=jnp.asarray(1, dtype=jnp.int32),
        ball_x=jnp.asarray(75, dtype=jnp.int32),
        ball_y=jnp.asarray(150, dtype=jnp.int32),
        ball_vel_x=jnp.asarray(-1.0, dtype=jnp.float32),
        ball_vel_y=jnp.asarray(0.0, dtype=jnp.float32),
    )

    slowed_enemy = JaxPong()._enemy_step(state)
    matched_enemy = env._enemy_step(state)

    assert int(slowed_enemy.enemy_y) == 100
    # Match-speed restores native 2 px/frame and bypasses selected cadence.
    assert int(matched_enemy.enemy_y) == 102
    assert int(matched_enemy.enemy_speed) == 2


def test_match_ball_speed_uses_accelerated_trajectory_not_just_velocity():
    env = PongAccelModWrapper(
        AtariWrapper(JaxPong(), noop_max=0, sticky_actions=0.0, first_fire=False),
        level_encoding="parameter_vector",
        enemy_match_ball_speed=True,
        parameter_delta_multipliers=(1.0, 4.0),
        parameter_enemy_move_intervals=(4,),
    )
    _, state = env.reset(jax.random.PRNGKey(3))
    core_state = _get_core_env_state(state).replace(
        step_counter=jnp.asarray(1, dtype=jnp.int32),
        enemy_y=jnp.asarray(24, dtype=jnp.int32),
        enemy_step_size=jnp.asarray(1, dtype=jnp.int32),
        enemy_move_interval=jnp.asarray(4, dtype=jnp.int32),
        ball_x=jnp.asarray(17, dtype=jnp.int32),
        ball_y=jnp.asarray(150, dtype=jnp.int32),
        # The raw Pong velocity is only one pixel.  The active BX level makes
        # its actual position delta four pixels, which must still defeat the
        # slow enemy cadence.
        ball_vel_x=jnp.asarray(-1.0, dtype=jnp.float32),
        ball_vel_y=jnp.asarray(1.0, dtype=jnp.float32),
    )
    state = _set_core_env_state(state, core_state)
    fast_ball_level = env.parameter_baseline_level.at[0].set(1)

    _, next_state, *_ = env.step(
        state,
        jnp.asarray(0, dtype=jnp.int32),
        fast_ball_level,
    )
    next_core_state = _get_core_env_state(next_state)

    assert bool(next_core_state.enemy_response_uses_accel_trajectory)
    assert float(next_core_state.enemy_response_ball_vel_x) == -4.0
    assert int(next_core_state.enemy_y) == 143
    assert float(next_core_state.ball_vel_x) == 1.0


def test_parameter_wrapper_exposes_enemy_cadence_and_fast_ball_options():
    env = PongAccelModWrapper(
        AtariWrapper(JaxPong(), noop_max=0),
        level_encoding="parameter_vector",
        enemy_match_ball_speed=True,
    )

    assert env.parameter_bin_counts.shape == (12,)
    assert int(env.parameter_baseline_level[-1]) == 0
    assert env._core_env.consts.ENEMY_MATCH_BALL_SPEED

    _, state = env.reset(jax.random.PRNGKey(2))
    cadence_four_level = env.parameter_baseline_level.at[-1].set(2)
    _, state, *_ = env.step(state, jnp.asarray(0, dtype=jnp.int32), cadence_four_level)

    assert int(_get_core_env_state(state).enemy_move_interval) == 4


def test_parameter_wrapper_can_omit_new_enemy_cadence_dimension():
    env = PongAccelModWrapper(
        AtariWrapper(_core_without_renderer(), noop_max=0),
        level_encoding="parameter_vector",
        enable_lazy_enemy=True,
        enable_enemy_move_interval=False,
        parameter_delta_multipliers=(1, 1.05, 1.1, 1.15, 1.2, 1.25),
        parameter_paddle_heights=(14, 15, 16),
        parameter_paddle_widths=(2, 3, 4),
        parameter_enemy_step_sizes=(1, 2, 3, 4),
        parameter_enemy_tracking_modes=(0, 1),
    )

    assert env.parameter_bin_counts.shape == (11,)
    assert "enemy_move_interval" not in env.accel_parameter_names
    assert all(
        spec.target != "enemy_move_interval" for spec in env._pre_step_specs
    )
    assert int(jnp.prod(env.parameter_bin_counts)) == 20_155_392


def test_lazy_enemy_is_a_configurable_adversarial_training_mod():
    env = PongAccelModWrapper(
        AtariWrapper(JaxPong(), noop_max=0, sticky_actions=0.0, first_fire=False),
        level_encoding="parameter_vector",
        enable_lazy_enemy=True,
    )
    assert env._pre_step_specs[-2].label == "lazy_enemy"
    assert env.parameter_bin_counts.shape == (12,)

    _, state = env.reset(jax.random.PRNGKey(4))
    core_state = _get_core_env_state(state).replace(
        step_counter=jnp.asarray(1, dtype=jnp.int32),
        enemy_y=jnp.asarray(100, dtype=jnp.int32),
        ball_y=jnp.asarray(120, dtype=jnp.int32),
        ball_vel_x=jnp.asarray(1.0, dtype=jnp.float32),
    )
    state = _set_core_env_state(state, core_state)
    lazy_enemy_level = env.parameter_baseline_level.at[-2].set(1)
    _, next_state, *_ = env.step(
        state,
        jnp.asarray(0, dtype=jnp.int32),
        lazy_enemy_level,
    )
    next_core_state = _get_core_env_state(next_state)

    assert int(next_core_state.enemy_tracking_mode) == 1
    assert int(next_core_state.enemy_y) == 100

    disabled_env = PongAccelModWrapper(
        AtariWrapper(JaxPong(), noop_max=0, sticky_actions=0.0, first_fire=False),
        level_encoding="parameter_vector",
        enable_lazy_enemy=False,
    )
    assert disabled_env.parameter_bin_counts.shape == (11,)
    assert all(spec.label != "lazy_enemy" for spec in disabled_env._pre_step_specs)
