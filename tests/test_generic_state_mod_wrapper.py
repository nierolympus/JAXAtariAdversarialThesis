import jax

from jaxatari.games.jax_kangaroo import JaxKangaroo
from jaxatari.games.jax_pong import JaxPong
from jaxatari.wrappers import GenericModSpec, GenericStateModWrapper


def test_generic_wrapper_can_unpack_constants_for_pong_and_kangaroo():
    pong_wrapper = GenericStateModWrapper(JaxPong(), mod_specs=[])
    kangaroo_wrapper = GenericStateModWrapper(JaxKangaroo(), mod_specs=[])

    pong_consts = pong_wrapper.get_constants_dict()
    kangaroo_consts = kangaroo_wrapper.get_constants_dict()

    assert "PADDLE_MAX_SPEED" in pong_consts
    assert "MIN_BALL_SPEED" in pong_consts
    assert "MOVEMENT_SPEED" in kangaroo_consts
    assert "LEVEL_1" in kangaroo_consts


def test_generic_wrapper_can_override_constants_on_core_env():
    env = GenericStateModWrapper(JaxPong(), mod_specs=[])
    env.set_constant_overrides({"PADDLE_MAX_SPEED": 9})

    assert float(env._core_env.consts.PADDLE_MAX_SPEED) == 9.0


def test_generic_wrapper_modifies_nested_kangaroo_state_fields():
    base_env = JaxKangaroo()
    env = GenericStateModWrapper(
        base_env,
        mod_specs=[GenericModSpec(target="player.vel_x", op="add", value=2)],
        info_fields={"modded_vel_x": "player.vel_x"},
    )

    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    # Compare no-op adversary action with active adversary action from the same state.
    _, state_no_mod, _, _, _ = env.step(state, 0, 0)
    _, state_mod, _, _, info = env.step(state, 0, 1)

    assert int(state_mod.player.vel_x) == int(state_no_mod.player.vel_x) + 2
    assert int(info["mod_action"]) == 1
    assert int(info["modded_vel_x"]) == int(state_mod.player.vel_x)

