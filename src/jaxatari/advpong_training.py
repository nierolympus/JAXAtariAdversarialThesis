import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.wrappers import PongStateModWrapper,FlattenObservationWrapper, ObjectCentricWrapper, AtariWrapper


def _to_int(x):
    return int(jax.device_get(x))


def _state_equal(a, b):
    return bool(jax.tree_util.tree_all(jax.tree.map(jnp.array_equal, a, b)))


def _print_key_fields(prefix, state):
    print(
        f"{prefix} player_speed={_to_int(state.player_speed):3d} "
        f"enemy_speed={_to_int(state.enemy_speed):3d} "
        f"acc={_to_int(state.acceleration_counter):3d} "
        f"ball_vx={_to_int(state.ball_vel_x):3d} "
        f"ball_vy={_to_int(state.ball_vel_y):3d}"
    )


n_steps = 20
rng = jax.random.PRNGKey(0)

base_env = jaxatari.make("pong")
adv_env =PongStateModWrapper(jaxatari.make("pong"))

# --- A) Base vs Adv(mod=0) should stay identical ---
_, base_state = base_env.reset(rng)
_, adv_state = adv_env.reset(rng)

print("A) Compare Base env vs Adv env with adversary_action=0")
print("Initial states equal:", _state_equal(base_state, adv_state))

actions = jax.random.randint(rng, (n_steps,), 0, base_env.action_space().n)
all_equal = True

for t in range(n_steps):
    a_t = actions[t]

    _, base_state, _, _, _ = base_env.step(base_state, a_t)
    _, adv_state, _, _, info = adv_env.step(adv_state, a_t, jnp.array(0, dtype=jnp.int32))

    same = _state_equal(base_state, adv_state)
    all_equal = all_equal and same

    if not same:
        print(f"First mismatch at step {t}, mod_action={_to_int(info['mod_action'])}")
        _print_key_fields("base:", base_state)
        _print_key_fields("adv0:", adv_state)
        break

print("PASS (states equal for all steps with mod=0):", all_equal)

# --- B) Base vs Adv(mod!=0) should diverge ---
_, base_state = base_env.reset(rng)
_, adv_state = adv_env.reset(rng)

# Force a positive velocity so mod 8 shows 1 -> 2 (instead of crossing -1 -> 1).
base_state = base_state.replace(ball_vel_x=jnp.array(1, dtype=jnp.int32))
adv_state = adv_state.replace(ball_vel_x=jnp.array(1, dtype=jnp.int32))

mod_action = jnp.array(8, dtype=jnp.int32)  # modifies ball_vel_x
print("\nB) Compare Base env vs Adv env with adversary_action=8")
print("Expected example: base ball_vx 1 -> 1, adv ball_vx 1 -> 2")

diverged = False
for t in range(n_steps):
    a_t = actions[t]

    _, base_state, _, _, _ = base_env.step(base_state, a_t)
    _, adv_state, _, _, info = adv_env.step(adv_state, a_t, mod_action)

    same = _state_equal(base_state, adv_state)
    if not same:
        diverged = True
        print(f"First divergence at step {t}, mod_action={_to_int(info['mod_action'])}")
        _print_key_fields("base:", base_state)
        _print_key_fields("adv8:", adv_state)
        break

print("PASS (states diverge with mod!=0):", diverged)

if not all_equal or not diverged:
    raise SystemExit(1)
