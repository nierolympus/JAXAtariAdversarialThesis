import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import JaxPong
from jaxatari.wrappers import PongStateModWrapper


def _clip_signed(value, limit: int) -> int:
    return int(jnp.clip(value, -limit, limit))


def _clip_ball_velocity(value, min_speed: int, max_speed: int) -> int:
    sign = -1 if int(value) < 0 else 1
    magnitude = int(jnp.clip(jnp.abs(value), min_speed, max_speed))
    return sign * magnitude


def test_pong_state_mod_wrapper_changes_target_fields():
    env = PongStateModWrapper(JaxPong())
    key = jax.random.PRNGKey(0)

    _, state = env.reset(key)
    base_n = env._env.action_space().n
    base_action = 0

    # Wrapper should keep the base agent action space unchanged.
    assert env.action_space().n == base_n

    _, no_mod_state, _, _, _ = env.step(state, base_action, 0)

    _, plus_player_state, _, _, _ = env.step(state, base_action, 2)
    _, plus_enemy_state, _, _, _ = env.step(state, base_action, 4)
    _, plus_acc_state, _, _, _ = env.step(state, base_action, 6)
    _, plus_vx_state, _, _, _ = env.step(state, base_action, 8)
    _, minus_vy_state, _, _, info = env.step(state, base_action, 9)

    max_speed = int(env._env.consts.MAX_SPEED)
    min_ball_speed = int(env._env.consts.MIN_BALL_SPEED)
    max_ball_speed = int(env._env.consts.BALL_MAX_SPEED)
    max_acc = len(env._env.consts.PLAYER_ACCELERATION) - 1

    assert int(plus_player_state.player_speed) == _clip_signed(no_mod_state.player_speed + 1, max_speed)
    assert int(plus_enemy_state.enemy_speed) == _clip_signed(no_mod_state.enemy_speed + 1, max_speed)
    assert int(plus_acc_state.acceleration_counter) == int(jnp.clip(no_mod_state.acceleration_counter + 1, 0, max_acc))
    assert int(plus_vx_state.ball_vel_x) == _clip_ball_velocity(no_mod_state.ball_vel_x + 1, min_ball_speed, max_ball_speed)
    assert int(minus_vy_state.ball_vel_y) == _clip_ball_velocity(no_mod_state.ball_vel_y - 1, min_ball_speed, max_ball_speed)

    assert int(info["mod_action"]) == 9
    assert int(info["player_speed"]) == int(minus_vy_state.player_speed)
    assert int(info["enemy_speed"]) == int(minus_vy_state.enemy_speed)
    assert int(info["acceleration_counter"]) == int(minus_vy_state.acceleration_counter)
    assert int(info["ball_vel_x"]) == int(minus_vy_state.ball_vel_x)
    assert int(info["ball_vel_y"]) == int(minus_vy_state.ball_vel_y)


def test_pong_state_mod_wrapper_keeps_legacy_dict_action_compatibility():
    env = PongStateModWrapper(JaxPong())
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)

    _, state_new_api, _, _, info_new_api = env.step(state, 0, 8)
    _, state_legacy, _, _, info_legacy = env.step(state, {"agent": 0, "adversary": 8})

    assert int(state_new_api.ball_vel_x) == int(state_legacy.ball_vel_x)
    assert int(info_new_api["mod_action"]) == int(info_legacy["mod_action"]) == 8

