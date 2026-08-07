import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import JaxPong


def test_dynamic_player_paddle_height_changes_render_and_collision():
    env = JaxPong()
    _, state = env.reset(jax.random.PRNGKey(0))

    # At this y coordinate, the ball misses the baseline 16px paddle but
    # intersects the 28px morphology paddle.
    baseline = state.replace(
        player_y=jnp.float32(100),
        ball_x=jnp.int32(139),
        ball_y=jnp.int32(125),
        ball_vel_x=jnp.float32(1),
        ball_vel_y=jnp.float32(0),
    )
    enlarged = baseline.replace(player_paddle_height=jnp.int32(28))

    baseline_after = env._ball_step(baseline, jnp.int32(0))
    enlarged_after = env._ball_step(enlarged, jnp.int32(0))

    assert float(baseline_after.ball_vel_x) > 0
    assert float(enlarged_after.ball_vel_x) < 0

    # Render away from the ball so its pixels cannot cover the paddle.
    baseline_render_state = baseline.replace(ball_x=jnp.int32(70))
    enlarged_render_state = enlarged.replace(ball_x=jnp.int32(70))
    player_colour = jnp.asarray(env.consts.PLAYER_COLOR, dtype=jnp.uint8)

    def paddle_pixels(render_state):
        frame = env.render(render_state)
        column = frame[:, env.consts.PLAYER_X, :]
        return jnp.sum(jnp.all(column == player_colour, axis=-1))

    assert int(paddle_pixels(baseline_render_state)) == 16
    assert int(paddle_pixels(enlarged_render_state)) == 28
