import jax
import jax.numpy as jnp

from jaxatari.environment import JAXAtariAction as Action
from jaxatari.games.jax_pong import JaxPong
from jaxatari.games.jax_pong_default import JaxPong as DefaultJaxPong


_REFERENCE_STATE_FIELDS = (
    "player_y",
    "player_speed",
    "ball_x",
    "ball_y",
    "enemy_y",
    "enemy_speed",
    "ball_vel_x",
    "ball_vel_y",
    "player_score",
    "enemy_score",
    "step_counter",
    "key",
)


def _assert_reference_state_matches(modified_state, reference_state):
    for field in _REFERENCE_STATE_FIELDS:
        assert bool(
            jnp.array_equal(
                getattr(modified_state, field), getattr(reference_state, field)
            )
        ), field


def _assert_observation_matches(modified_obs, reference_obs):
    modified_leaves = jax.tree_util.tree_leaves(modified_obs)
    reference_leaves = jax.tree_util.tree_leaves(reference_obs)

    assert len(modified_leaves) == len(reference_leaves)
    for modified_leaf, reference_leaf in zip(modified_leaves, reference_leaves):
        assert bool(jnp.array_equal(modified_leaf, reference_leaf))


def test_default_jax_pong_matches_reference_trajectory_and_rendering():
    modified_env = JaxPong()
    reference_env = DefaultJaxPong()
    key = jax.random.PRNGKey(42)

    modified_obs, modified_state = modified_env.reset(key)
    reference_obs, reference_state = reference_env.reset(key)

    _assert_reference_state_matches(modified_state, reference_state)
    _assert_observation_matches(modified_obs, reference_obs)
    assert int(modified_state.player_paddle_height) == 16
    assert int(modified_state.player_paddle_width) == 4
    assert int(modified_state.enemy_step_size) == 2
    assert int(modified_state.enemy_tracking_mode) == 0
    assert int(modified_state.enemy_move_interval) == 0
    assert bool(jnp.array_equal(modified_env.render(modified_state), reference_env.render(reference_state)))

    # Exercise each compact Pong action.  With no ACCEL option enabled, every
    # base-game field and observation must follow the reference implementation.
    for action in (0, 2, 2, 0, 3, 3, 0, 4, 0, 5) * 4:
        modified_out = modified_env.step(
            modified_state, jnp.asarray(action, dtype=jnp.int32)
        )
        reference_out = reference_env.step(
            reference_state, jnp.asarray(action, dtype=jnp.int32)
        )
        modified_obs, modified_state, modified_reward, modified_done, modified_info = modified_out
        reference_obs, reference_state, reference_reward, reference_done, reference_info = reference_out

        _assert_reference_state_matches(modified_state, reference_state)
        _assert_observation_matches(modified_obs, reference_obs)
        assert bool(jnp.array_equal(modified_reward, reference_reward))
        assert bool(jnp.array_equal(modified_done, reference_done))
        assert bool(jnp.array_equal(modified_info.time, reference_info.time))

    assert bool(jnp.array_equal(modified_env.render(modified_state), reference_env.render(reference_state)))


def test_ball_speed_cap_is_not_part_of_unwrapped_pong_physics():
    """The cap is now an ACCEL-wrapper safeguard, matching the reference game."""
    modified_env = JaxPong()
    reference_env = DefaultJaxPong()
    key = jax.random.PRNGKey(7)
    _, modified_state = modified_env.reset(key)
    _, reference_state = reference_env.reset(key)

    # A firing player returns a 4 px/frame ball at 5 px/frame.  Reference Pong
    # allows that boost; only an active non-unit ACCEL scale may clamp it.
    modified_state = modified_state.replace(
        ball_x=jnp.asarray(139, dtype=jnp.int32),
        ball_y=jnp.asarray(100, dtype=jnp.int32),
        ball_vel_x=jnp.asarray(4.0, dtype=jnp.float32),
        ball_vel_y=jnp.asarray(0.0, dtype=jnp.float32),
    )
    reference_state = reference_state.replace(
        ball_x=jnp.asarray(139, dtype=jnp.int32),
        ball_y=jnp.asarray(100, dtype=jnp.int32),
        ball_vel_x=jnp.asarray(4, dtype=jnp.int32),
        ball_vel_y=jnp.asarray(0, dtype=jnp.int32),
    )

    modified_after = modified_env._ball_step(
        modified_state, jnp.asarray(Action.FIRE, dtype=jnp.int32)
    )
    reference_after = reference_env._ball_step(
        reference_state, jnp.asarray(Action.FIRE, dtype=jnp.int32)
    )

    assert float(modified_after.ball_vel_x) == -5.0
    assert bool(jnp.array_equal(modified_after.ball_vel_x, reference_after.ball_vel_x))
