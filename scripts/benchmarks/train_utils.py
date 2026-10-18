import os
from typing import Dict, Union
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict


def log_wandb_video(
    frames,
    label,
    env_step,
    fps=30,
    caption=None,
    total_reward=None,
    metrics_payload=None,
):
    """Log one rollout using the shared ACCEL/non-ACCEL W&B schema."""
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim == 3:
        frames = frames[..., None]
    if frames.ndim != 4:
        raise ValueError(
            f"Expected video frames shaped (T, H, W, C), got {frames.shape}."
        )
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    elif frames.shape[-1] >= 3:
        frames = frames[..., :3]
    else:
        raise ValueError(
            "Video frames must have one or at least three channels, "
            f"got {frames.shape}."
        )

    host_step = int(np.asarray(env_step))
    video_payload = {
        "num_env_steps": host_step,
        f"video/{label}": wandb.Video(
            np.transpose(frames, (0, 3, 1, 2)),
            fps=fps,
            format="mp4",
            caption=caption,
        ),
    }
    shared_metrics = dict(metrics_payload or {})
    shared_metrics["num_env_steps"] = host_step
    shared_metrics.setdefault(f"video_metrics/{label}/length", len(frames))
    if total_reward is not None:
        shared_metrics.setdefault(
            f"video_metrics/{label}/return", float(total_reward)
        )

    wandb.log(video_payload, step=host_step)
    wandb.log(shared_metrics, step=host_step)


def evaluation_video_callback(
    states,
    dones,
    rewards,
    step,
    periodic_due,
    final_due,
    *,
    renderer,
    eval_label,
    eval_every_steps,
    fps=30,
):
    """Render a plain-PQN eval rollout with the ACCEL W&B video layout."""
    print("Rendering evaluation video...")
    while True:
        if hasattr(states, "atari_state"):
            states = states.atari_state
        elif hasattr(states, "env_state"):
            states = states.env_state
        else:
            break

    done_values = np.asarray(dones, dtype=bool).reshape(-1)
    done_indices = np.flatnonzero(done_values)
    available_states = jax.tree_util.tree_leaves(states)[0].shape[0]
    num_states = (
        int(done_indices[0]) + 1 if len(done_indices) else int(available_states)
    )
    if num_states <= 0:
        return

    states = jax.tree_util.tree_map(lambda value: value[:num_states], states)
    frames = np.asarray(jax.vmap(renderer.render)(states), dtype=np.uint8)
    total_reward = float(
        np.asarray(rewards, dtype=np.float32).reshape(-1)[:num_states].sum()
    )
    host_step = int(np.asarray(step))
    safe_label = str(eval_label).replace("/", "_").replace(" ", "_")
    video_prefixes = []
    if bool(np.asarray(periodic_due)):
        evaluation_index = max(1, host_step // max(int(eval_every_steps), 1))
        video_prefixes.append(f"eval_{evaluation_index:03d}")
    if bool(np.asarray(final_due)):
        video_prefixes.append("final")

    for prefix in video_prefixes:
        log_wandb_video(
            frames,
            label=f"{prefix}/{safe_label}",
            env_step=host_step,
            fps=fps,
            caption="Evaluation rollout without an ACCEL level.",
            total_reward=total_reward,
        )
    print("Evaluation video done.")


def video_callback(
    states,
    dones,
    step,
    renderer,
    mod=False,
    mod_name=None,
    mod_actions=None,
    adversary_video=False,
    fps=30,
    step_metric=None,
):
    """
    Starting a new thread to render video so that it doesn't block training.
    """
    # video_thread = threading.Thread(target=video_renderer, args=(states, dones, step, renderer, mod))
    # video_thread.start()
    video_renderer(
        states,
        dones,
        step,
        renderer,
        mod,
        mod_name,
        mod_actions,
        adversary_video,
        fps,
        step_metric,
    )

def video_renderer(
    states,
    dones,
    step,
    renderer,
    mod,
    mod_name,
    mod_actions,
    adversary_video,
    fps=30,
    step_metric=None,
):
    print("Rendering video...")
    video_folder = f"{wandb.run.dir}/media/videos/"
    os.makedirs(video_folder, exist_ok=True)

    while hasattr(states, 'atari_state'):
        states = states.atari_state

    if hasattr(states, 'env_state'):
        states = states.env_state

    # num_states is where the first done is True
    num_states = jnp.argmax(dones)
    # or len of the first array of the states pytree
    if num_states == 0:
        num_states = jax.tree_util.tree_leaves(states)[0].shape[0]

    # select every 4th frame (and only the first num_states)
    states_reduced = jax.tree_util.tree_map(lambda x: x[:num_states], states)
    rasters = jax.vmap(renderer.render)(states_reduced)
    frames = np.array(rasters, dtype=np.uint8)

    if adversary_video and mod_actions is not None:
        actions = np.array(mod_actions)
        if actions.ndim > 1:
            actions = actions[:, 0]
        actions = actions[:num_states]
        max_action = max(int(np.max(actions)) if len(actions) else 0, 1)
        active = actions != 0

        # Add a visible top strip: gray=no adversary action, colored=adversary action id.
        palette = np.array(
            [
                [80, 80, 80],
                [230, 57, 70],
                [29, 185, 84],
                [69, 123, 157],
                [255, 183, 3],
                [131, 56, 236],
                [251, 86, 7],
                [0, 180, 216],
            ],
            dtype=np.uint8,
        )
        colors = palette[actions.astype(np.int32) % len(palette)]
        frames[:, :4, :, :] = colors[:, None, None, :]

        action_counts = {
            f"adversary_action_{int(action)}": int(count)
            for action, count in zip(*np.unique(actions, return_counts=True))
        }
        wandb.log(
            {
                f"adversary_video_active_fraction_{step}": float(active.mean()) if len(active) else 0.0,
                f"adversary_video_max_action_{step}": max_action,
                f"adversary_video_action_counts_{step}": action_counts,
            }
        )
    # shape currently is (N, W, H, 3)
    # but should be (N, 3, W, H)
    frames = np.transpose(frames, (0, 3, 1, 2))

    video = wandb.Video(frames, fps=fps, format="mp4")
    name = f"video_{step}"
    if mod:
        if mod_name is not None:
            safe_mod_name = str(mod_name).replace("/", "_")
            name = f"video_{step}_mod_{safe_mod_name}"
        else:
            name = f"video_{step}_mod"
    if adversary_video:
        name = f"{name}_adversary"
    payload = {name: video}
    if step_metric is None:
        wandb.log(payload)
    else:
        host_step = int(np.asarray(step))
        payload[step_metric] = host_step
        wandb.log(payload, step=host_step)
    print("Video done.")

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")
