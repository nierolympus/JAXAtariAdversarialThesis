import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import JaxPong, PongConstants


def test_default_episode_reset_keeps_reference_vertical_serve_direction():
    env = JaxPong()
    keys = jax.random.split(jax.random.PRNGKey(0), 64)

    velocities = jax.vmap(lambda key: env.reset(key)[1].ball_vel_y)(keys)

    assert set(jax.device_get(velocities).tolist()) == {1.0}


def test_default_post_goal_reset_keeps_reference_vertical_serve_direction():
    env = JaxPong()
    _, state = env.reset(jax.random.PRNGKey(1))

    from_above = env._score_and_reset(
        state.replace(
            ball_x=jnp.asarray(157, dtype=jnp.int32),
            ball_y=jnp.asarray(80, dtype=jnp.int32),
        )
    )
    from_below = env._score_and_reset(
        state.replace(
            ball_x=jnp.asarray(157, dtype=jnp.int32),
            ball_y=jnp.asarray(140, dtype=jnp.int32),
        )
    )

    assert float(from_above.ball_vel_y) == -1.0
    assert float(from_below.ball_vel_y) == 1.0


def test_randomized_serves_are_an_explicit_core_option():
    env = JaxPong(PongConstants(RANDOMIZE_SERVE_DIRECTION=True))
    keys = jax.random.split(jax.random.PRNGKey(2), 64)

    initial_velocities = jax.vmap(lambda key: env.reset(key)[1].ball_vel_y)(keys)

    def score_and_get_serve(key):
        _, state = env.reset(key)
        state = state.replace(
            ball_x=jnp.asarray(157, dtype=jnp.int32),
            ball_y=jnp.asarray(80, dtype=jnp.int32),
        )
        return env._score_and_reset(state).ball_vel_y

    post_goal_velocities = jax.vmap(score_and_get_serve)(keys)

    assert set(jax.device_get(initial_velocities).tolist()) == {-1.0, 1.0}
    assert set(jax.device_get(post_goal_velocities).tolist()) == {-1.0, 1.0}
