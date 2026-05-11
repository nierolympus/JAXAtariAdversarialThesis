import os

import jax
import jax.numpy as jnp
import numpy as np
import pygame

import jaxatari
from jaxatari.environment import JAXAtariAction
from jaxatari.wrappers import AtariWrapper, GenericModSpec, GenericStateModWrapper


def get_human_action() -> int:
    """Returns the JAXAtariAction constant for the currently pressed key."""
    key_map = {
        pygame.K_UP: JAXAtariAction.UP,
        pygame.K_DOWN: JAXAtariAction.DOWN,
        pygame.K_LEFT: JAXAtariAction.LEFT,
        pygame.K_RIGHT: JAXAtariAction.RIGHT,
        pygame.K_SPACE: JAXAtariAction.FIRE,
    }
    pressed_keys = pygame.key.get_pressed()
    for key, action in key_map.items():
        if pressed_keys[key]:
            return action
    return JAXAtariAction.NOOP


def render_modified_pong() -> None:
    """Interactive Pong demo with mod actions and simple frame recording."""
    print("\nStarting interactive PyGame render test...")
    print("Press '1' to multiply the enemy speed by 1.5.")
    print("Press '2' to multiply the ball_vel_x by 1.5.")
    print("Press '0' to apply no modification.")
    print("Use arrow keys to move the player paddle.")
    print("\n[INFO] ball_vel_x/ball_vel_y are now stored as float32 in PongState.")
    print("       Float multipliers are preserved (no int truncation).\n")

    mod_specs = [
        GenericModSpec(
            target="ball_vel_y",
            op="mul",
            value=1.5,
            preserve_sign=True,
            dtype=jnp.float32,
        ),
        GenericModSpec(
            target="ball_vel_x",
            op="mul",
            value=1.5,
            min_value=-4.0,
            max_value=4.0,
            preserve_sign=True,
            dtype=jnp.float32,
        ),
    ]

    base_env = jaxatari.make("pong")
    atari_env = AtariWrapper(base_env)
    env = GenericStateModWrapper(atari_env, mod_specs=mod_specs)

    renderer = env._core_env.renderer

    pygame.init()
    rng = jax.random.PRNGKey(42)
    dummy_frame = renderer.render(env._core_env.reset(rng)[1])
    screen_size = (dummy_frame.shape[1] * 3, dummy_frame.shape[0] * 3)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Interactive Pong Test")
    clock = pygame.time.Clock()

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(env._core_env.render)

    # Debug helpers (keep step separate to inspect mod vs. env effects)
    jitted_apply_mod = jax.jit(env._apply_mod)
    jitted_env_step = jax.jit(env._env.step)

    def map_action_to_index(action_constant):
        action_set = jnp.array(env.ACTION_SET)
        action_int = int(action_constant)
        matches = jnp.where(action_set == action_int)[0]
        return jnp.array(matches[0] if matches.size > 0 else 0, dtype=jnp.int32)

    rng = jax.random.PRNGKey(42)
    _, state = jitted_reset(rng)
    mod_action = jnp.array(0, dtype=jnp.int32)
    apply_mod_once = False

    record_frames = True
    record_every_n = 1
    max_record_frames = 3000
    recorded_frames = []
    recorded_count = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    mod_action = jnp.array(0, dtype=jnp.int32)
                    apply_mod_once = False
                    print("Mod Action: 0 (None)")
                elif event.key == pygame.K_1:
                    mod_action = jnp.array(1, dtype=jnp.int32)
                    apply_mod_once = True
                    print("Mod Action: 1 (Enemy Speed x1.5)")
                elif event.key == pygame.K_2:
                    mod_action = jnp.array(2, dtype=jnp.int32)
                    apply_mod_once = True
                    print("Mod Action: 2 (Ball Vel X x1.5)")
                elif event.key == pygame.K_r:
                    print("Resetting environment.")
                    _, state = jitted_reset(rng)

        human_action_constant = get_human_action()
        action = map_action_to_index(human_action_constant)

        step_mod_action = mod_action if apply_mod_once else jnp.array(0, dtype=jnp.int32)
        pre_mod_vel = float(state.env_state.ball_vel_x)
        mod_state = jitted_apply_mod(state, step_mod_action)
        post_mod_vel = float(mod_state.env_state.ball_vel_x)

        _, state, _, terminated, truncated, info = jitted_env_step(
            mod_state, action
        )
        post_step_vel = float(state.env_state.ball_vel_x)

        if apply_mod_once:
            apply_mod_once = False

        # Debug: Print state before/after mod and after step
        if step_mod_action != 0:
            ball_vel_y = float(state.env_state.ball_vel_y)
            print(
                f"[MOD {int(step_mod_action)}] ball_vel_x pre={pre_mod_vel}, "
                f"post_mod={post_mod_vel}, post_step={post_step_vel}, "
                f"ball_vel_y={ball_vel_y}"
            )

        frame = jitted_render(state.env_state)
        cpu_frame = np.asarray(frame)
        if (
            record_frames
            and (recorded_count % record_every_n == 0)
            and len(recorded_frames) < max_record_frames
        ):
            recorded_frames.append(cpu_frame)
        recorded_count += 1

        frame_surface = pygame.surfarray.make_surface(cpu_frame.transpose(1, 0, 2))
        scaled_surface = pygame.transform.scale(frame_surface, screen_size)
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print("Episode finished. Resetting.")
            _, state = jitted_reset(rng)

        clock.tick(60)

    pygame.quit()
    if record_frames and recorded_frames:
        np.savez_compressed(
            record_path,
            frames=np.stack(recorded_frames, axis=0),
            fps=60,
        )
        print(f"[PASS] PyGame render test finished. Saved frames to {record_path}")
    else:
        print("[PASS] PyGame render test finished.")


def main() -> None:
    render_modified_pong()


if __name__ == "__main__":
    main()

