import jax
import jax.numpy as jnp

from jaxatari.accel_wrappers import FreewayAccelModWrapper
from jaxatari.core import GAME_MODULES
from jaxatari.environment import JaxEnvironment
from jaxatari.games.jax_freeway import FreewayConstants, FreewayRenderer, JaxFreeway
from jaxatari.games.jax_freeway_default import (
    FreewayConstants as DefaultFreewayConstants,
)
from jaxatari.games.jax_freeway_default import JaxFreeway as DefaultJaxFreeway
from jaxatari.games.mods.freeway_mods import FreewayEnvMod
from jaxatari.rendering import jax_rendering_utils as render_utils
from jaxatari.wrappers import AtariWrapper, _get_core_env_state


def _make_without_renderer(env_type, constants_type):
    """Construct Freeway physics without loading optional sprite assets."""
    env = object.__new__(env_type)
    JaxEnvironment.__init__(env, constants_type())
    return env


def test_neutral_freeway_controls_match_default_car_trajectory():
    env = _make_without_renderer(JaxFreeway, FreewayConstants)
    reference_env = _make_without_renderer(
        DefaultJaxFreeway, DefaultFreewayConstants
    )

    with jax.disable_jit():
        _, state = env.reset(jax.random.PRNGKey(0))
        _, reference_state = reference_env.reset(jax.random.PRNGKey(0))
        for action in (0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 0, 0):
            _, state, reward, done, _ = env.step(state, jnp.asarray(action))
            _, reference_state, reference_reward, reference_done, _ = (
                reference_env.step(reference_state, jnp.asarray(action))
            )

            assert bool(jnp.array_equal(state.cars, reference_state.cars))
            assert bool(jnp.array_equal(state.chicken_y, reference_state.chicken_y))
            assert bool(jnp.array_equal(state.score, reference_state.score))
            assert bool(jnp.array_equal(state.time, reference_state.time))
            assert bool(jnp.allclose(state.lane_time, reference_state.lane_time))
            assert int(reward) == int(reference_reward)
            assert bool(done) == bool(reference_done)


def test_freeway_speed_and_size_fields_drive_physics_and_observation():
    env = _make_without_renderer(JaxFreeway, FreewayConstants)

    with jax.disable_jit():
        _, state = env.reset(jax.random.PRNGKey(1))
        state = state.replace(
            cars=state.cars.at[:, 0].set(80),
            lane_time=jnp.zeros_like(state.lane_time),
            car_speed_multiplier=jnp.asarray(2.0, dtype=jnp.float32),
            car_width=jnp.asarray(12, dtype=jnp.int32),
            car_height=jnp.asarray(14, dtype=jnp.int32),
        )
        obs, state, _, _, _ = env.step(state, jnp.asarray(0))

    # Lane four has native period one and moves left, so 2x speed moves 2 px.
    assert int(state.cars[4, 0]) == 78
    assert bool(jnp.all(obs.car.width == 12))
    assert bool(jnp.all(obs.car.height == 14))


def test_freeway_accel_parameter_vector_and_eval_mod_registry():
    core_env = _make_without_renderer(JaxFreeway, FreewayConstants)
    env = FreewayAccelModWrapper(
        AtariWrapper(
            core_env,
            sticky_actions=0.0,
            episodic_life=False,
            first_fire=False,
            noop_max=0,
        ),
        level_encoding="parameter_vector",
    )

    assert env.accel_parameter_names == (
        "car_speed_multiplier",
        "car_width",
        "car_height",
    )
    assert tuple(map(int, env.parameter_baseline_level)) == (0, 0, 0)

    with jax.disable_jit():
        _, state = env.reset(jax.random.PRNGKey(2))
        values = env.decode_parameter_level(jnp.asarray([2, 2, 2]))
        modified_state = env._apply_pre_step_parameters(state, values)
        core_state = _get_core_env_state(modified_state)

    assert tuple(map(float, values)) == (2.0, 12.0, 14.0)
    assert float(core_state.car_speed_multiplier) == 2.0
    assert int(core_state.car_width) == 12
    assert int(core_state.car_height) == 14
    assert "freeway_default" in GAME_MODULES
    assert "slow_cars" in FreewayEnvMod.REGISTRY
    assert "stop_all_cars" in FreewayEnvMod.REGISTRY


def test_dynamic_car_renderer_preserves_native_pixels_and_resizes():
    renderer = object.__new__(FreewayRenderer)
    renderer.config = render_utils.RendererConfig(
        game_dimensions=(30, 30), channels=3, downscale=None
    )
    renderer.jr = render_utils.JaxRenderingUtils(renderer.config)
    raster = jnp.zeros((30, 30), dtype=jnp.int32)
    car_mask = jnp.ones((10, 8), dtype=jnp.int32)

    with jax.disable_jit():
        native = renderer._render_car_at_size(
            raster, 3, 4, car_mask, jnp.asarray(8), jnp.asarray(10)
        )
        expected = renderer.jr.render_at_clipped(raster, 3, 4, car_mask)
        enlarged = renderer._render_car_at_size(
            raster, 3, 4, car_mask, jnp.asarray(12), jnp.asarray(14)
        )

    assert bool(jnp.array_equal(native, expected))
    assert int(jnp.count_nonzero(native)) == 80
    assert int(jnp.count_nonzero(enlarged)) == 12 * 14


if __name__ == "__main__":
    test_neutral_freeway_controls_match_default_car_trajectory()
    test_freeway_speed_and_size_fields_drive_physics_and_observation()
    test_freeway_accel_parameter_vector_and_eval_mod_registry()
    test_dynamic_car_renderer_preserves_native_pixels_and_resizes()
    print("Freeway ACCEL smoke tests passed.")
