import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import JaxPong


def test_enemy_tracking_mode_can_reproduce_lazy_enemy_gate():
    env = JaxPong()
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        step_counter=jnp.asarray(1, dtype=jnp.int32),
        enemy_y=jnp.asarray(100, dtype=jnp.int32),
        ball_y=jnp.asarray(120, dtype=jnp.int32),
        enemy_step_size=jnp.asarray(2, dtype=jnp.int32),
    )

    continuous = env._enemy_step(
        state.replace(
            enemy_tracking_mode=jnp.asarray(0, dtype=jnp.int32),
            ball_vel_x=jnp.asarray(1.0, dtype=jnp.float32),
        )
    )
    approach_only_away = env._enemy_step(
        state.replace(
            enemy_tracking_mode=jnp.asarray(1, dtype=jnp.int32),
            ball_vel_x=jnp.asarray(1.0, dtype=jnp.float32),
        )
    )
    approach_only_toward = env._enemy_step(
        state.replace(
            enemy_tracking_mode=jnp.asarray(1, dtype=jnp.int32),
            ball_vel_x=jnp.asarray(-1.0, dtype=jnp.float32),
        )
    )

    assert int(continuous.enemy_y) == 102
    assert int(approach_only_away.enemy_y) == 100
    assert int(approach_only_toward.enemy_y) == 102
