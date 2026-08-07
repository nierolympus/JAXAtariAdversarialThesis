import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import JaxPong
from jaxatari.wrappers import (
    AccelDeltaModWrapper,
    AtariWrapper,
    LevelStateModSpec,
    _get_core_env_state,
)


def test_pre_step_morphology_mask_sets_and_restores_paddle_height():
    env = AccelDeltaModWrapper(
        AtariWrapper(JaxPong(), noop_max=0),
        mod_specs=[],
        pre_step_specs=[
            LevelStateModSpec(
                target="player_paddle_height",
                baseline_const="PLAYER_PADDLE_BASE_HEIGHT",
                op="mul",
                value=1.75,
                min_const="PLAYER_PADDLE_MIN_HEIGHT",
                max_const="PLAYER_PADDLE_MAX_HEIGHT",
                dtype=jnp.int32,
            )
        ],
    )
    _, state = env.reset(jax.random.PRNGKey(0))

    _, state, _, _, _, info = env.step(
        state,
        jnp.int32(0),
        jnp.array([True]),
    )
    assert int(_get_core_env_state(state).player_paddle_height) == 28
    assert int(info["morphology/player_paddle_height"]) == 28

    _, state, _, _, _, _ = env.step(
        state,
        jnp.int32(0),
        jnp.array([False]),
    )
    assert int(_get_core_env_state(state).player_paddle_height) == 16
