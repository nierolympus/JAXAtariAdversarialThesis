import jax
import jax.numpy as jnp

from jaxatari.accel_wrappers import SeaquestAccelModWrapper
from jaxatari.core import GAME_MODULES
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.environment import JaxEnvironment
from jaxatari.games.jax_seaquest import (
    JaxSeaquest,
    SeaquestConstants,
)
from jaxatari.games.jax_seaquest_default import (
    JaxSeaquest as DefaultJaxSeaquest,
)
from jaxatari.games.jax_seaquest_default import (
    SeaquestConstants as DefaultSeaquestConstants,
)
from jaxatari.wrappers import (
    AtariWrapper,
    _get_core_env_state,
    _set_core_env_state,
)


_REFERENCE_STATE_FIELDS = (
    "player_x",
    "player_y",
    "player_direction",
    "oxygen",
    "divers_collected",
    "score",
    "lives",
    "spawn_state",
    "diver_positions",
    "shark_positions",
    "sub_positions",
    "enemy_missile_positions",
    "surface_sub_position",
    "player_missile_position",
    "step_counter",
    "just_surfaced",
    "successful_rescues",
    "death_counter",
    "rng_key",
)


def _make_without_renderer(env_type, constants_type):
    """Construct Seaquest logic without requiring optional sprite assets."""
    env = object.__new__(env_type)
    JaxEnvironment.__init__(env, constants_type())
    return env


def _assert_tree_equal(actual, expected, label):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves), label
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        assert bool(jnp.array_equal(actual_leaf, expected_leaf)), label


def _assert_reference_state_equal(modified_state, reference_state):
    for field in _REFERENCE_STATE_FIELDS:
        _assert_tree_equal(
            getattr(modified_state, field),
            getattr(reference_state, field),
            field,
        )


def test_neutral_seaquest_velocity_controls_match_default_trajectory():
    modified_env = _make_without_renderer(JaxSeaquest, SeaquestConstants)
    reference_env = _make_without_renderer(
        DefaultJaxSeaquest, DefaultSeaquestConstants
    )
    key = jax.random.PRNGKey(42)

    with jax.disable_jit():
        modified_obs, modified_state = modified_env.reset(key)
        reference_obs, reference_state = reference_env.reset(key)

        assert int(modified_state.player_move_speed) == 1
        assert int(modified_state.enemy_speed_multiplier) == 1
        assert int(modified_state.player_missile_speed) == 5
        assert int(modified_state.enemy_missile_speed_multiplier) == 1
        _assert_reference_state_equal(modified_state, reference_state)
        _assert_tree_equal(modified_obs, reference_obs, "reset observation")

        # Start underwater with full oxygen so this short trace exercises
        # movement immediately instead of the initial oxygen-refill freeze.
        modified_state = modified_state.replace(
            player_y=jnp.asarray(80), oxygen=jnp.asarray(64)
        )
        reference_state = reference_state.replace(
            player_y=jnp.asarray(80), oxygen=jnp.asarray(64)
        )
        for action in (3, 2, 1, 0, 4, 5):
            modified_out = modified_env.step(modified_state, jnp.asarray(action))
            reference_out = reference_env.step(reference_state, jnp.asarray(action))
            (
                modified_obs,
                modified_state,
                modified_reward,
                modified_done,
                modified_info,
            ) = modified_out
            (
                reference_obs,
                reference_state,
                reference_reward,
                reference_done,
                reference_info,
            ) = reference_out

            _assert_reference_state_equal(modified_state, reference_state)
            _assert_tree_equal(modified_obs, reference_obs, "observation")
            _assert_tree_equal(modified_reward, reference_reward, "reward")
            _assert_tree_equal(modified_done, reference_done, "done")
            _assert_tree_equal(modified_info, reference_info, "info")


def test_each_seaquest_velocity_field_changes_its_physics_path():
    env = _make_without_renderer(JaxSeaquest, SeaquestConstants)

    with jax.disable_jit():
        _, state = env.reset(jax.random.PRNGKey(7))

        faster_player = state.replace(
            player_x=jnp.asarray(76),
            player_y=jnp.asarray(80),
            player_move_speed=jnp.asarray(2),
        )
        player_x, _, _ = env.player_step(faster_player, jnp.asarray(Action.RIGHT))
        assert int(player_x) == 78

        faster_torpedo = state.replace(
            player_y=jnp.asarray(80),
            player_missile_position=jnp.asarray([50, 87, 1]),
            player_missile_speed=jnp.asarray(7),
        )
        player_missile = env.player_missile_step(
            faster_torpedo,
            jnp.asarray(76),
            jnp.asarray(80),
            jnp.asarray(Action.NOOP),
        )
        assert int(player_missile[0]) == 57

        sharks = state.shark_positions.at[0].set(jnp.asarray([50, 71, 1]))
        moved_sharks, _, _, _ = env.step_enemy_movement(
            state.spawn_state,
            sharks,
            state.sub_positions,
            jnp.asarray(0),
            jnp.asarray(2),
            state.rng_key,
        )
        assert int(moved_sharks[0, 0]) == 52

        enemy_missiles = state.enemy_missile_positions.at[0].set(
            jnp.asarray([50, 73, 1])
        )
        moved_enemy_missiles = env.enemy_missiles_step(
            state.sub_positions,
            enemy_missiles,
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(3),
        )
        assert int(moved_enemy_missiles[0, 0]) == 53

        surface_state = state.replace(
            surface_sub_position=jnp.asarray([100, 45, -1]),
            step_counter=jnp.asarray(4),
            enemy_speed_multiplier=jnp.asarray(2),
        )
        assert int(env.surface_sub_step(surface_state)[0]) == 98


def test_seaquest_accel_wrapper_decodes_and_sets_velocity_vector():
    core_env = _make_without_renderer(JaxSeaquest, SeaquestConstants)
    env = SeaquestAccelModWrapper(
        AtariWrapper(
            core_env,
            sticky_actions=0.0,
            episodic_life=False,
            first_fire=False,
            noop_max=0,
        ),
        level_encoding="parameter_vector",
        parameter_player_move_speeds=(1, 2),
        parameter_enemy_speed_multipliers=(1, 2),
        parameter_player_missile_speeds=(3, 5, 7),
        parameter_enemy_missile_speed_multipliers=(1, 2),
    )

    assert env.accel_parameter_names == (
        "player_move_speed",
        "enemy_speed_multiplier",
        "player_missile_speed",
        "enemy_missile_speed_multiplier",
    )
    assert tuple(map(int, env.parameter_baseline_level)) == (0, 0, 1, 0)

    with jax.disable_jit():
        _, state = env.reset(jax.random.PRNGKey(11))
        values = env.decode_parameter_level(jnp.asarray([1, 1, 0, 1]))
        modified_state = env._apply_pre_step_parameters(state, values)
        core_state = _get_core_env_state(modified_state)

    assert tuple(map(int, values)) == (2, 2, 3, 2)
    assert int(core_state.player_move_speed) == 2
    assert int(core_state.enemy_speed_multiplier) == 2
    assert int(core_state.player_missile_speed) == 3
    assert int(core_state.enemy_missile_speed_multiplier) == 2

    # Exercise the normal wrapper step path as well: the selected speed is in
    # the core state before Seaquest handles movement and collisions.
    prepared_core = _get_core_env_state(state).replace(
        player_x=jnp.asarray(76),
        player_y=jnp.asarray(80),
        oxygen=jnp.asarray(64),
    )
    prepared_state = _set_core_env_state(state, prepared_core)
    with jax.disable_jit():
        _, stepped_state, _, _, _, info = env.step(
            prepared_state,
            jnp.asarray(3),  # Compact Seaquest RIGHT action.
            jnp.asarray([1, 1, 0, 1]),
        )
    stepped_core = _get_core_env_state(stepped_state)
    assert int(stepped_core.player_x) == 78
    assert int(info["morphology/player_move_speed"]) == 2


def test_seaquest_default_clone_is_registered():
    assert GAME_MODULES["seaquest"] == "jaxatari.games.jax_seaquest"
    assert (
        GAME_MODULES["seaquest_default"]
        == "jaxatari.games.jax_seaquest_default"
    )
