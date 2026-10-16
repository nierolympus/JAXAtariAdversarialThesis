import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxatari.accel_wrappers import KangarooAccelWrapper
from jaxatari.environment import JaxEnvironment
from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooConstants
from jaxatari.games.mods.kangaroo_mods import KangarooEnvMod
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    _get_core_env_state,
)


def _atari(core_env):
    return AtariWrapper(
        core_env,
        sticky_actions=0.0,
        episodic_life=False,
        first_fire=False,
        noop_max=0,
    )


def _core_without_renderer():
    """Construct game logic without requiring optional sprite assets."""
    env = object.__new__(JaxKangaroo)
    JaxEnvironment.__init__(env, KangarooConstants())
    env.obs_size = 111
    return env


def _assert_tree_equal(actual, expected):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        assert bool(jnp.array_equal(actual_leaf, expected_leaf))


def _move_first_fruit(level, x):
    return level._replace(
        fruit_positions=level.fruit_positions.at[0, 0, 0].set(x)
    )


def test_kangaroo_mutation_changes_only_one_collectible_x():
    env = KangarooAccelWrapper(_atari(_core_without_renderer()), mutation_edits=1)
    baseline = env.baseline_level

    with jax.disable_jit():
        mutated = env.mutate_level(jax.random.PRNGKey(3), baseline)

    _assert_tree_equal(mutated.platform_positions, baseline.platform_positions)
    _assert_tree_equal(mutated.platform_sizes, baseline.platform_sizes)
    _assert_tree_equal(mutated.ladder_sizes, baseline.ladder_sizes)
    _assert_tree_equal(mutated.ladder_positions, baseline.ladder_positions)
    _assert_tree_equal(
        mutated.fruit_positions[..., 1], baseline.fruit_positions[..., 1]
    )
    _assert_tree_equal(
        mutated.bell_positions[..., 1], baseline.bell_positions[..., 1]
    )

    fruit_changed = (
        mutated.fruit_positions[..., 0] != baseline.fruit_positions[..., 0]
    )
    bell_changed = (
        mutated.bell_positions[..., 0] != baseline.bell_positions[..., 0]
    )
    assert int(jnp.sum(fruit_changed) + jnp.sum(bell_changed)) == 1
    assert bool(jnp.all(mutated.fruit_positions[..., 0] >= env._object_min_x))
    assert bool(jnp.all(mutated.fruit_positions[..., 0] <= env._fruit_max_x))
    assert bool(jnp.all(mutated.bell_positions[..., 0] >= env._object_min_x))
    assert bool(jnp.all(mutated.bell_positions[..., 0] <= env._bell_max_x))


def test_kangaroo_live_collectible_positions_use_structured_level():
    core_env = _core_without_renderer()
    layout = _move_first_fruit(core_env.get_structured_level(), 64)

    with jax.disable_jit():
        state = core_env.apply_structured_level(core_env.reset_level(1), layout)
        state = state.replace(
            player=state.player.replace(x=jnp.asarray(64), y=jnp.asarray(108))
        )
        fruit_score, fruit_actives, _, _ = core_env._fruits_step(state)

    assert int(state.level.fruit_positions[0, 0]) == 64
    assert int(fruit_score) == 100
    assert not bool(fruit_actives[0])


def test_kangaroo_live_collision_geometry_uses_structured_level():
    core_env = _core_without_renderer()
    baseline = core_env.get_structured_level()
    layout = baseline._replace(
        ladder_positions=baseline.ladder_positions.at[0, 0, 0].set(64)
    )

    with jax.disable_jit():
        state = core_env.apply_structured_level(core_env.reset_level(1), layout)
        state = state.replace(
            player=state.player.replace(x=jnp.asarray(64), y=jnp.asarray(108))
        )
        ladder_hits = core_env._player_is_above_ladder(state)

    assert bool(ladder_hits[0])
    assert int(state.level.ladder_positions[0, 0]) == 64


def test_kangaroo_platform_physics_reads_live_structured_geometry():
    core_env = _core_without_renderer()
    baseline = core_env.get_structured_level()
    layout = baseline._replace(
        platform_positions=baseline.platform_positions.at[0, 0, 0].set(100),
        platform_sizes=baseline.platform_sizes.at[0, 0, 0].set(20),
    )

    with jax.disable_jit():
        state = core_env.apply_structured_level(core_env.reset_level(1), layout)
        platforms_below = core_env._get_platforms_below_player(state)

    assert not bool(jnp.any(platforms_below))


def test_kangaroo_baseline_wrapper_matches_unmodified_atari_step():
    core_env = _core_without_renderer()
    plain_env = _atari(core_env)
    accel_env = KangarooAccelWrapper(_atari(core_env))
    key = jax.random.PRNGKey(11)

    with jax.disable_jit():
        plain_obs, plain_state = plain_env.reset(key)
        accel_obs, accel_state = accel_env.reset(key, accel_env.baseline_level)
        _assert_tree_equal(accel_obs, plain_obs)
        _assert_tree_equal(accel_state, plain_state)

        for action in (0, 3, 3, 0, 2, 0):
            plain_step = plain_env.step(plain_state, jnp.asarray(action))
            accel_step = accel_env.step(
                accel_state,
                jnp.asarray(action),
                accel_env.baseline_level,
                jnp.asarray(1.0),
            )
            _assert_tree_equal(accel_step, plain_step)
            plain_state = plain_step[1]
            accel_state = accel_step[1]


def test_runner_observation_stack_forwards_structured_reset_level():
    accel_env = KangarooAccelWrapper(_atari(_core_without_renderer()))
    env = LogWrapper(
        NormalizeObservationWrapper(
            FlattenObservationWrapper(
                ObjectCentricWrapper(accel_env, frame_skip=1)
            )
        )
    )
    level = _move_first_fruit(accel_env.baseline_level, 64)

    with jax.disable_jit():
        _, state = env.reset(jax.random.PRNGKey(19), level)

    core_state = _get_core_env_state(state)
    assert int(core_state.level.fruit_positions[0, 0]) == 64


def test_second_level_only_observation_matches_declared_shape():
    core_env = _core_without_renderer()
    modded_env = KangarooEnvMod(core_env, mods_config=["second_level_only"])
    env = NormalizeObservationWrapper(
        FlattenObservationWrapper(ObjectCentricWrapper(_atari(modded_env)))
    )

    reset_keys = jax.random.split(jax.random.PRNGKey(29), 2)
    compiled_reset = jax.jit(jax.vmap(env.reset)).lower(reset_keys).compile()
    obs, state = compiled_reset(reset_keys)

    core_state = _get_core_env_state(state)
    assert bool(jnp.all(core_state.current_level == 2))
    assert core_state.level.platform_positions.shape == (2, 20, 2)
    assert core_state.level.ladder_positions.shape == (2, 20, 2)
    assert obs.shape == (2, *env.observation_space().shape) == (2, 1760)


def test_jaxued_level_sampler_stores_structured_kangaroo_pytrees():
    jaxued_src = Path(__file__).parents[1] / "jaxued" / "jaxued" / "src"
    if str(jaxued_src) not in sys.path:
        sys.path.insert(0, str(jaxued_src))
    from jaxued.level_sampler import LevelSampler

    env = KangarooAccelWrapper(_atari(_core_without_renderer()))
    baseline = env.baseline_level
    sampler_api = LevelSampler(capacity=4, duplicate_check=True)

    with jax.disable_jit():
        mutated = env.mutate_level(jax.random.PRNGKey(23), baseline)
        sampler = sampler_api.initialize(baseline)
        sampler, baseline_index = sampler_api.insert(
            sampler, baseline, jnp.asarray(1.0)
        )
        sampler, mutated_index = sampler_api.insert(
            sampler, mutated, jnp.asarray(2.0)
        )
        replayed = sampler_api.get_levels(sampler, mutated_index)

    assert int(sampler["size"]) == 2
    assert int(baseline_index) == 0
    assert int(mutated_index) == 1
    _assert_tree_equal(replayed, mutated)
