import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    PongStateModWrapper,
)


def _state_equal(a, b) -> bool:
    return bool(jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, a, b)))


def _build_chain(kind: str):
    """Build wrapper chains with Generic/Pong wrapper as the last wrapper."""
    base = jaxatari.make("pong")
    atari = AtariWrapper(base)
    if kind == "atari":
        inner = atari
    elif kind == "object_flatten":
        inner = FlattenObservationWrapper(ObjectCentricWrapper(atari))
    elif kind == "pixel":
        # Pixel wrapper must be applied directly after AtariWrapper.
        inner = PixelObsWrapper(atari)
    else:
        raise ValueError(f"Unknown chain kind: {kind}")
    return PongStateModWrapper(inner)


def _smoke_chain(kind: str) -> None:
    env = _build_chain(kind)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)
    action = jnp.array(0, dtype=jnp.int32)
    mod_action = jnp.array(8, dtype=jnp.int32)
    _, next_state, reward, terminated, truncated, info = env.step(state, action, mod_action)
    assert obs is not None
    assert next_state is not None
    assert reward is not None
    assert terminated is not None
    assert truncated is not None
    assert "mod_action" in info


def _mod0_equal_and_mod8_diverge() -> None:
    """On pure Atari chain: mod=0 stays equal, mod!=0 diverges."""
    base_env = AtariWrapper(jaxatari.make("pong"))
    adv_env = PongStateModWrapper(AtariWrapper(jaxatari.make("pong")))
    rng = jax.random.PRNGKey(42)
    actions = jax.random.randint(rng, (12,), 0, base_env.action_space().n)

    # A) mod_action=0 -> equal trajectories
    _, s_base = base_env.reset(rng)
    _, s_adv = adv_env.reset(rng)
    for a in actions:
        _, s_base, _, _, _, _ = base_env.step(s_base, a)
        _, s_adv, _, _, _, _ = adv_env.step(s_adv, a, jnp.array(0, dtype=jnp.int32))
        assert _state_equal(s_base, s_adv), "State mismatch while mod_action=0"

    # B) mod_action=8 -> should diverge
    _, s_base = base_env.reset(rng)
    _, s_adv = adv_env.reset(rng)
    s_base = s_base.replace(env_state=s_base.env_state.replace(ball_vel_x=jnp.array(1, dtype=jnp.int32)))
    s_adv = s_adv.replace(env_state=s_adv.env_state.replace(ball_vel_x=jnp.array(1, dtype=jnp.int32)))

    diverged = False
    for a in actions:
        _, s_base, _, _, _, _ = base_env.step(s_base, a)
        _, s_adv, _, _, _, _ = adv_env.step(s_adv, a, jnp.array(8, dtype=jnp.int32))
        if not _state_equal(s_base, s_adv):
            diverged = True
            break
    assert diverged, "Expected divergence for mod_action=8"


def main() -> None:
    chains = ["atari", "object_flatten", "pixel"]
    for kind in chains:
        _smoke_chain(kind)
        print(f"[PASS] smoke chain: {kind}")

    _mod0_equal_and_mod8_diverge()
    print("[PASS] mod_action=0 equal and mod_action=8 diverges (atari chain)")


if __name__ == "__main__":
    main()
