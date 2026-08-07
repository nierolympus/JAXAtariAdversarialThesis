"""
PQN + JAXUED-style PLR/ACCEL for JAXAtari adversarial levels.

This runner keeps the JAXAtari wrapper workflow from ACCEL.py, but replaces
the host-side ACCELLevelBuffer with JAXUED's LevelSampler and the scheduling
logic used by jaxued/examples/maze_plr.py.

Level meaning in this file:
    Maze PLR level       -> full maze layout
    JAXAtari level here  -> either a bool adversarial mask or a discrete
                            parameter vector, shape (num_level_dimensions,)

ACCEL mode follows Maze PLR's branch schedule:
    0: new random levels
    1: replay sampled levels from LevelSampler
    2: mutate the last replayed levels and insert/update them
"""

from __future__ import annotations

import copy
import os
import re
import sys
import time
from enum import IntEnum
from math import comb
from pathlib import Path
from typing import Any, Sequence, Tuple

import hydra
import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxatari.wrappers import (
    AccelDeltaModWrapper,
    AtariWrapper,
    FlattenObservationWrapper,
    GenericModSpec,
    LevelStateModSpec,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
)
from omegaconf import OmegaConf

from pqn_agent_adv_random import QNetwork, Transition
from train_utils import save_params


REPO_ROOT = Path(__file__).resolve().parents[2]
JAXUED_SRC = REPO_ROOT / "jaxued" / "jaxued" / "src"
if str(JAXUED_SRC) not in sys.path:
    sys.path.insert(0, str(JAXUED_SRC))

from jaxued.level_sampler import LevelSampler  # noqa: E402
from jaxued.linen import ResetRNN  # noqa: E402
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss  # noqa: E402


class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class CustomTrainState(TrainState):
    batch_stats: Any
    sampler: Any
    update_state: Any
    last_replay_level_batch: Any
    dr_last_level_batch: Any
    mutation_last_level_batch: Any
    current_level_batch: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    num_dr_updates: int = 0
    num_replay_updates: int = 0
    num_mutation_updates: int = 0


class PongAccelModWrapper(AccelDeltaModWrapper):
    """Pong ACCEL with mask or discrete parameter-vector levels.

    The first seven specs amplify a physics-caused state delta every step.  In
    particular, ``ball_x`` and ``ball_y`` make the ball visibly travel faster;
    their amplified movement is capped at the game's physical ball-speed
    limit. Parameter levels then set paddle height, paddle width, and enemy
    tracking directly; legacy masks expose separate grow/shrink height bits.
    """

    def __init__(
        self,
        env,
        player_paddle_height_multiplier: float = 1.75,
        player_paddle_shrink_multiplier: float = 0.5,
        player_paddle_width_multiplier: float = 1.5,
        enemy_step_multiplier: float = 2.0,
        max_ball_trajectory_step: float = 4.0,
        level_encoding: str = "mask",
        parameter_delta_multipliers: Sequence[float] = (
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
        ),
        parameter_paddle_heights: Sequence[float] = (8, 10, 12, 14, 16, 20, 24, 28, 32),
        parameter_paddle_widths: Sequence[float] = (2, 3, 4, 5, 6, 7, 8),
        parameter_enemy_step_sizes: Sequence[float] = (1, 2, 3, 4),
    ):
        delta_specs = [
            # BX: amplify the horizontal ball displacement each raw Pong step.
            # The cap prevents ACCEL from moving the ball faster than 4 px/step.
            GenericModSpec(
                target="ball_x",
                min_value=0,
                max_value=160,
                max_abs_delta=max_ball_trajectory_step,
                dtype=jnp.int32,
            ),
            # BY: amplify vertical ball displacement, including wall-bounce
            # travel, with the same physical 4 px/step cap.
            GenericModSpec(
                target="ball_y",
                min_value=24,
                max_value=194,
                max_abs_delta=max_ball_trajectory_step,
                dtype=jnp.int32,
            ),
            # VX: amplify changes to horizontal ball velocity caused by paddle
            # hits/boosts; the velocity itself remains in [-4, +4].
            GenericModSpec(
                target="ball_vel_x",
                op="mul",
                value=10,
                min_const="MIN_BALL_SPEED",
                max_const="MAX_BALL_SPEED",
                preserve_sign=True,
                dtype=jnp.float32,
            ),
            # VY: amplify vertical deflection changes caused by a paddle or
            # wall bounce; its magnitude is also bounded by the Pong limit.
            GenericModSpec(
                target="ball_vel_y",
                op="mul",
                value=1,
                min_const="MIN_BALL_SPEED",
                max_const="MAX_BALL_SPEED",
                preserve_sign=True,
                dtype=jnp.float32,
            ),
            # PY: amplify the player's physics-generated vertical displacement
            # while keeping the paddle inside the court.
            GenericModSpec(
                target="player_y",
                min_const="PADDLE_MIN_Y",
                max_value=190,
                max_abs_delta=5.75,
                dtype=jnp.float32,
            ),
            # EY: amplify the enemy AI paddle's vertical tracking movement,
            # capped at four pixels per raw step.
            GenericModSpec(
                target="enemy_y",
                min_value=24,
                max_value=190,
                max_abs_delta=4,
                dtype=jnp.int32,
            ),
            # PS: amplify the player's analog velocity change. This changes
            # acceleration/deceleration, not the paddle's collision geometry.
            GenericModSpec(
                target="player_speed",
                min_value=-5.75,
                max_value=5.75,
                dtype=jnp.float32,
            ),
        ]
        morphology_specs = [
            # PH: before every raw Pong physics step, replace the normal 16 px
            # player height with 16 * 1.75 = 28 px. This is recomputed from
            # baseline each step, so it changes rendering, movement bounds,
            # observations, and collision height without compounding.
            LevelStateModSpec(
                target="player_paddle_height",
                label="player_paddle_height_grow",
                baseline_const="PLAYER_PADDLE_BASE_HEIGHT",
                op="mul",
                value=player_paddle_height_multiplier,
                min_const="PLAYER_PADDLE_MIN_HEIGHT",
                max_const="PLAYER_PADDLE_MAX_HEIGHT",
                dtype=jnp.int32,
            ),
            # SH: before every raw Pong physics step, replace the normal 16 px
            # player height with 16 * 0.5 = 8 px. It is mutually exclusive
            # with PH, so a sampled level can make the paddle larger or
            # smaller, never both at once.
            LevelStateModSpec(
                target="player_paddle_height",
                label="player_paddle_height_shrink",
                baseline_const="PLAYER_PADDLE_BASE_HEIGHT",
                op="mul",
                value=player_paddle_shrink_multiplier,
                min_const="PLAYER_PADDLE_MIN_HEIGHT",
                max_const="PLAYER_PADDLE_MAX_HEIGHT",
                dtype=jnp.int32,
            ),
            # PW: before every raw Pong physics step, replace the normal 4 px
            # paddle width with 4 * 1.5 = 6 px. This changes the rendered
            # width, object observation width, and horizontal hit interval.
            LevelStateModSpec(
                target="player_paddle_width",
                baseline_const="PLAYER_PADDLE_BASE_WIDTH",
                op="mul",
                value=player_paddle_width_multiplier,
                min_const="PLAYER_PADDLE_MIN_WIDTH",
                max_const="PLAYER_PADDLE_MAX_WIDTH",
                dtype=jnp.int32,
            ),
            # ES: before each raw Pong step, replace the enemy's normal 2 px
            # AI tracking increment with 2 * 2 = 4 px. The enemy still moves
            # on its normal schedule (seven of every eight raw frames); each
            # scheduled movement is simply larger.
            LevelStateModSpec(
                target="enemy_step_size",
                baseline_const="ENEMY_STEP_SIZE",
                op="mul",
                value=enemy_step_multiplier,
                min_const="ENEMY_STEP_MIN_SIZE",
                max_const="ENEMY_STEP_MAX_SIZE",
                dtype=jnp.int32,
            ),
        ]
        # A parameter-vector level owns one absolute paddle-height dimension;
        # the mask mode needs separate grow/shrink bits for backwards
        # compatibility. Keeping only one height spec here avoids two vector
        # coordinates writing the same field in sequence.
        parameter_mode = str(level_encoding).lower() == "parameter_vector"
        pre_step_specs = (
            [morphology_specs[0], morphology_specs[2], morphology_specs[3]]
            if parameter_mode
            else morphology_specs
        )
        super().__init__(
            env,
            mod_specs=delta_specs,
            pre_step_specs=pre_step_specs,
            level_encoding=level_encoding,
            # The seven delta dimensions independently scale the raw
            # physics-caused deltas. The remaining four set absolute values
            # before collision detection: height, width, enemy step size.
            parameter_values=(
                *([parameter_delta_multipliers] * len(delta_specs)),
                parameter_paddle_heights,
                parameter_paddle_widths,
                parameter_enemy_step_sizes,
            ),
        )
        if not parameter_mode:
            # PH and SH write the same height field in opposite directions.
            # Keep them mutually exclusive during mask-level generation.
            height_grow_index = len(delta_specs)
            height_shrink_index = height_grow_index + 1
            self._conflicting_mask_pairs = (
                (height_grow_index, height_shrink_index),
                (height_shrink_index, height_grow_index),
            )


class AtariRNNActorCritic(nn.Module):
    """JAXUED-style recurrent actor-critic adapted to JAXAtari observations."""

    action_dim: int
    obs_shape: Sequence[int]
    rnn_hidden_size: int = 256
    hidden_size: int = 128

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        obs = obs.astype(jnp.float32)
        leading_dims = obs.shape[: -len(self.obs_shape)]
        x = obs.reshape((*leading_dims, -1))

        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="embed0")(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="embed1")(x)
        x = nn.relu(x)

        hidden, x = ResetRNN(nn.OptimizedLSTMCell(features=self.rnn_hidden_size))(
            (x, dones),
            initial_carry=hidden,
        )

        actor = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(x)
        actor = nn.relu(actor)
        actor = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor)
        pi = distrax.Categorical(logits=actor)

        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(x)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)


def eps_greedy_exploration(rng, q_vals, eps):
    rng_a, rng_e = jax.random.split(rng)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    random_actions = jax.random.randint(
        rng_a,
        shape=greedy_actions.shape,
        minval=0,
        maxval=q_vals.shape[-1],
    )
    return jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < eps,
        random_actions,
        greedy_actions,
    )


def initialize_actor_critic_carry(batch_dims, hidden_size: int):
    return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
        jax.random.PRNGKey(0),
        (*batch_dims, hidden_size),
    )


def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """JAXUED GAE helper, copied structurally from maze_plr.py."""

    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: CustomTrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, CustomTrainState], chex.ArrayTree]:
    """JAXUED PPO/RNN update adapted to the local TrainState type."""
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn({"params": params}, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                actor_loss = (-jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                critic_loss = 0.5 * jnp.maximum(
                    (values_pred - targets) ** 2,
                    (values_pred_clipped - targets) ** 2,
                ).mean()
                loss = actor_loss + critic_coeff * critic_loss - entropy_coeff * entropy
                return loss, (critic_loss, actor_loss, entropy)

            loss, grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            train_state = jax.lax.cond(
                update_grad,
                lambda ts: ts.apply_gradients(grads=grads),
                lambda ts: ts,
                train_state,
            )
            train_state = train_state.replace(grad_steps=train_state.grad_steps + update_grad)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


def compute_per_env_regret(chosen_q, lambda_targets, dones):
    """Positive value loss per env, ignoring samples after an episode boundary."""
    prev_done = jnp.concatenate(
        [jnp.zeros((1, dones.shape[1]), dtype=bool), dones[:-1]],
        axis=0,
    )
    valid = ~prev_done
    per_step_loss = jnp.maximum(0.0, lambda_targets - chosen_q)
    sum_loss = jnp.sum(per_step_loss * valid, axis=0)
    count_valid = jnp.sum(valid, axis=0)
    return jnp.where(count_valid > 0, sum_loss / count_valid, 0.0)


def _mean_infos(infos, skip=()):
    metrics = {}
    for key, value in infos.items():
        if key in skip:
            continue
        metrics[key] = jnp.mean(value)
    return metrics


def _host_metrics(metrics):
    return {key: float(jax.device_get(value)) for key, value in metrics.items()}


def _safe_metric_fragment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


def _short_accel_label(target: str) -> str:
    """Return a compact in-video label for the Pong ACCEL action legend."""
    return {
        "ball_x": "BX",
        "ball_y": "BY",
        "ball_vel_x": "VX",
        "ball_vel_y": "VY",
        "player_y": "PY",
        "enemy_y": "EY",
        "player_speed": "PS",
        "player_paddle_height_grow": "PH",
        "player_paddle_height_shrink": "SH",
        "player_paddle_height": "PH",
        "player_paddle_width": "PW",
        "enemy_step_size": "ES",
    }.get(target, _safe_metric_fragment(target)[:2].upper())


def _overlay_accel_mask_banner(
    frames: np.ndarray,
    mask: np.ndarray,
    labels: Sequence[str],
    height: int = 24,
) -> None:
    """Burn a readable active/inactive ACCEL-mask legend into video frames.

    The overlay occupies the unused Pong score area above the court.  Green
    panels are active, charcoal panels are inactive, and each panel has a
    two-letter target label.  This deliberately lives in the video pixels so
    it remains visible when W&B displays only the media artifact.
    """
    if len(mask) != len(labels):
        raise ValueError("ACCEL mask and action-label lengths must match.")
    if not len(mask) or height <= 0:
        return

    banner_height = min(max(int(height), 8), frames.shape[1])
    edges = np.linspace(0, frames.shape[2], len(mask) + 1, dtype=int)
    active_color = np.array([25, 185, 84], dtype=np.uint8)
    inactive_color = np.array([42, 42, 42], dtype=np.uint8)
    separator_color = np.array([236, 236, 236], dtype=np.uint8)
    dim_text_color = np.array([150, 150, 150], dtype=np.uint8)

    # 3x5 bitmap glyphs keep the legend dependency-free and readable at the
    # native 160-pixel Pong width.
    glyphs = {
        "B": ("110", "101", "110", "101", "110"),
        "E": ("111", "100", "110", "100", "111"),
        "H": ("101", "101", "111", "101", "101"),
        "P": ("110", "101", "110", "100", "100"),
        "V": ("101", "101", "101", "101", "010"),
        "X": ("101", "101", "010", "101", "101"),
        "Y": ("101", "101", "010", "010", "010"),
        "?": ("110", "001", "010", "000", "010"),
    }
    pixel_scale = 2
    glyph_width = 3 * pixel_scale
    glyph_gap = pixel_scale

    for index, (is_active, label) in enumerate(zip(mask, labels)):
        left, right = edges[index], edges[index + 1]
        panel_color = active_color if is_active else inactive_color
        text_color = separator_color if is_active else dim_text_color
        frames[:, :banner_height, left:right, :] = panel_color

        # A white status line makes active panels unambiguous even for users
        # with limited color discrimination or a heavily compressed preview.
        if is_active:
            frames[:, banner_height - 3 : banner_height, left:right, :] = separator_color

        label = label[:2].upper().ljust(2, "?")
        label_width = len(label) * glyph_width + (len(label) - 1) * glyph_gap
        x = left + max((right - left - label_width) // 2, 0)
        y = max((banner_height - 5 * pixel_scale) // 2, 0)
        for character in label:
            glyph = glyphs.get(character, glyphs["?"])
            for row, row_bits in enumerate(glyph):
                for column, bit in enumerate(row_bits):
                    if bit == "1":
                        frames[
                            :,
                            y + row * pixel_scale : y + (row + 1) * pixel_scale,
                            x + column * pixel_scale : x + (column + 1) * pixel_scale,
                            :,
                        ] = text_color
            x += glyph_width + glyph_gap

    # Keep thin separators between panels so adjacent active panels do not
    # visually merge into a single modification.
    for edge in edges[1:-1]:
        frames[:, :banner_height, max(edge - 1, 0) : edge + 1, :] = separator_color


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _float_tuple_config(config, key: str, default: Sequence[float]) -> tuple[float, ...]:
    """Read a YAML list as a static float tuple suitable for JAX tracing."""
    return tuple(float(value) for value in config.get(key, default))


def _unwrap_state_for_render(state):
    """Reach the raw JAXAtari game state from the normal wrapper stack."""
    while True:
        if hasattr(state, "atari_state"):
            state = state.atari_state
        elif hasattr(state, "env_state"):
            state = state.env_state
        else:
            return state


def train_state_to_log_dict(train_state: CustomTrainState, level_sampler: LevelSampler) -> dict:
    """JAXUED-style lightweight sampler logging for mask levels."""
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    return {
        "level_sampler/size": sampler["size"],
        "level_sampler/episode_count": sampler["episode_count"],
        "level_sampler/max_score": sampler["scores"].max(),
        "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
        "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
    }


def _enforce_mask_conflicts(mask, conflicting_pairs: tuple[tuple[int, int], ...]):
    """Clear the second bit of each ordered mutually exclusive pair."""
    for kept_index, cleared_index in conflicting_pairs:
        mask = mask.at[cleared_index].set(
            mask[cleared_index] & ~mask[kept_index]
        )
    return mask


def _make_random_mask(
    rng,
    mod_action_space_n: int,
    max_level_size: int,
    conflicting_pairs: tuple[tuple[int, int], ...] = (),
):
    """Sample one bool mask with exactly k active bits, k uniform in [0, max]."""
    max_size = min(max_level_size, mod_action_space_n)
    rng_size, rng_perm = jax.random.split(rng)
    size = jax.random.randint(rng_size, shape=(), minval=0, maxval=max_size + 1)
    perm = jax.random.permutation(rng_perm, mod_action_space_n)
    selected = jnp.arange(mod_action_space_n) < size
    inv_perm_selected = jnp.zeros((mod_action_space_n,), dtype=bool).at[perm].set(selected)
    return _enforce_mask_conflicts(inv_perm_selected, conflicting_pairs)


def _mutate_mask(
    rng,
    mask,
    max_level_size: int,
    conflicting_pairs: tuple[tuple[int, int], ...] = (),
):
    """Add or remove one active mask bit, matching ACCEL's add/remove mutation."""
    n = mask.shape[0]
    mask = _enforce_mask_conflicts(mask, conflicting_pairs)
    active_count = jnp.sum(mask)
    add_eligible = ~mask
    for active_index, blocked_index in conflicting_pairs:
        add_eligible = add_eligible.at[blocked_index].set(
            add_eligible[blocked_index] & ~mask[active_index]
        )
    can_add = (active_count < max_level_size) & add_eligible.any()
    can_remove = active_count > 0

    rng_dir, rng_add, rng_remove = jax.random.split(rng, 3)
    do_add = jax.lax.select(
        can_add & can_remove,
        jax.random.bernoulli(rng_dir, 0.5),
        can_add,
    )

    add_scores = jax.random.uniform(rng_add, (n,))
    remove_scores = jax.random.uniform(rng_remove, (n,))
    add_idx = jnp.argmax(jnp.where(add_eligible, add_scores, -1.0))
    remove_idx = jnp.argmax(jnp.where(mask, remove_scores, -1.0))

    def add():
        return mask.at[add_idx].set(True)

    def remove():
        return mask.at[remove_idx].set(False)

    def unchanged():
        return mask

    return jax.lax.cond(
        can_add | can_remove,
        lambda: jax.lax.cond(do_add, add, remove),
        unchanged,
    )


def _make_random_parameter_level(rng, bin_counts, baseline_level, baseline_prob: float):
    """Sample a quantized level, occasionally preserving exact base Pong."""
    bin_counts = jnp.asarray(bin_counts, dtype=jnp.int32)
    baseline_level = jnp.asarray(baseline_level, dtype=jnp.int32)
    rng_baseline, rng_values = jax.random.split(rng)
    values = jax.vmap(
        lambda key, count: jax.random.randint(key, (), 0, count, dtype=jnp.int32)
    )(jax.random.split(rng_values, bin_counts.shape[0]), bin_counts)
    return jax.lax.select(
        jax.random.bernoulli(rng_baseline, baseline_prob), baseline_level, values
    )


def _mutate_parameter_level(
    rng,
    level,
    bin_counts,
    multi_param_probability: float,
):
    """Move one, or occasionally two, parameters by a single bounded bin."""
    level = jnp.asarray(level, dtype=jnp.int32)
    bin_counts = jnp.asarray(bin_counts, dtype=jnp.int32)
    n = level.shape[0]
    rng_perm, rng_signs, rng_multi = jax.random.split(rng, 3)
    indices = jax.random.permutation(rng_perm, n)
    signs = jax.random.choice(
        rng_signs, jnp.asarray([-1, 1], dtype=jnp.int32), shape=(2,)
    )
    mutate_second = jax.random.bernoulli(rng_multi, multi_param_probability)

    def mutate_one(values, index, sign):
        return values.at[index].set(
            jnp.clip(values[index] + sign, 0, bin_counts[index] - 1)
        )

    mutated = mutate_one(level, indices[0], signs[0])
    return jax.lax.cond(
        mutate_second,
        lambda values: mutate_one(values, indices[1], signs[1]),
        lambda values: values,
        mutated,
    )


def _compute_mask_scores(config, dones, rewards, chosen_q, lambda_targets):
    """Score adversarial masks using JAXUED UED score utilities where requested.

    The default keeps the dense Q-learning positive-value-loss proxy from the
    first implementation because Atari rollouts may not complete an episode in
    every NUM_STEPS window. Set JAXUED_SCORE_FUNCTION=pvl or max_mc to use the
    JAXUED utilities directly.
    """
    score_fn = str(config.get("JAXUED_SCORE_FUNCTION", "q_pvl_all_steps")).lower()
    incomplete_value = float(config.get("JAXUED_INCOMPLETE_SCORE", 0.0))

    if score_fn in {"pvl", "jaxued_pvl"}:
        return positive_value_loss(
            dones,
            lambda_targets - chosen_q,
            incomplete_value=incomplete_value,
        )
    if score_fn in {"max_mc", "maxmc"}:
        max_returns = compute_max_returns(dones, rewards)
        return max_mc(
            dones,
            chosen_q,
            max_returns,
            incomplete_value=incomplete_value,
        )
    if score_fn in {"q_pvl_all_steps", "dense_pvl"}:
        return compute_per_env_regret(chosen_q, lambda_targets, dones)
    raise ValueError(
        "JAXUED_SCORE_FUNCTION must be one of "
        "'q_pvl_all_steps', 'pvl', or 'max_mc'."
    )

def make_train_jaxatari_accel_plr(config):
    config = copy.deepcopy(config)
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = int(
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0

    alg_mode = str(config.get("JAXUED_ALG", "accel")).lower()
    if alg_mode not in {"accel", "plr"}:
        raise ValueError(f"JAXUED_ALG must be 'accel' or 'plr', got {alg_mode!r}")
    protagonist_alg = str(config.get("PROTAGONIST_ALG", "pqn")).lower()
    if protagonist_alg not in {"pqn", "jaxued_ppo"}:
        raise ValueError(f"PROTAGONIST_ALG must be 'pqn' or 'jaxued_ppo', got {protagonist_alg!r}")

    train_mods = config.get("TRAIN_MODS", None)
    train_mods_list = train_mods if isinstance(train_mods, list) else [train_mods] if train_mods else None

    env = jaxatari.make(config["ENV_NAME"].lower(), mods=train_mods_list)

    def apply_wrappers(base_env):
        wrapped = AtariWrapper(base_env)
        if config.get("OBJECT_CENTRIC", False):
            wrapped = ObjectCentricWrapper(wrapped)
            wrapped = FlattenObservationWrapper(wrapped)
        else:
            wrapped = PixelObsWrapper(
                wrapped,
                do_pixel_resize=config.get("PIXEL_RESIZE", True),
                pixel_resize_shape=tuple(config.get("PIXEL_RESIZE_SHAPE", [84, 84])),
                grayscale=config.get("PIXEL_GRAYSCALE", True),
                use_native_downscaling=config.get("USE_NATIVE_DOWNSCALING", True),
            )
        wrapped = NormalizeObservationWrapper(wrapped)
        wrapped = LogWrapper(wrapped)
        return wrapped

    wrapped_env = apply_wrappers(env)
    adv_wrapper = config.get("ADV_WRAPPER", "pong")
    if adv_wrapper.lower() != "pong":
        raise ValueError(f"This example currently supports ADV_WRAPPER='pong', got {adv_wrapper!r}")
    level_encoding = str(config.get("ACCEL_LEVEL_ENCODING", "mask")).lower()
    if level_encoding not in {"mask", "parameter_vector"}:
        raise ValueError(
            "ACCEL_LEVEL_ENCODING must be 'mask' or 'parameter_vector', got "
            f"{level_encoding!r}"
        )
    env = PongAccelModWrapper(
        wrapped_env,
        player_paddle_height_multiplier=float(
            config.get("ACCEL_PLAYER_PADDLE_HEIGHT_MULTIPLIER", 1.75)
        ),
        player_paddle_shrink_multiplier=float(
            config.get("ACCEL_PLAYER_PADDLE_SHRINK_MULTIPLIER", 0.5)
        ),
        player_paddle_width_multiplier=float(
            config.get("ACCEL_PLAYER_PADDLE_WIDTH_MULTIPLIER", 1.5)
        ),
        enemy_step_multiplier=float(config.get("ACCEL_ENEMY_STEP_MULTIPLIER", 2.0)),
        max_ball_trajectory_step=float(
            config.get("ACCEL_MAX_BALL_TRAJECTORY_STEP", 4.0)
        ),
        level_encoding=level_encoding,
        parameter_delta_multipliers=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_DELTA_MULTIPLIERS",
            (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5),
        ),
        parameter_paddle_heights=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_PADDLE_HEIGHTS",
            (8, 10, 12, 14, 16, 20, 24, 28, 32),
        ),
        parameter_paddle_widths=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_PADDLE_WIDTHS",
            (2, 3, 4, 5, 6, 7, 8),
        ),
        parameter_enemy_step_sizes=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_ENEMY_STEP_SIZES",
            (1, 2, 3, 4),
        ),
    )

    eval_during_training = bool(
        config.get("EVAL_DURING_TRAINING", config.get("TEST_DURING_TRAINING", False))
    )
    # A final snapshot is useful even when periodic evaluation is disabled;
    # it represents the completed protagonist rather than an intermediate one.
    final_eval = bool(config.get("FINAL_EVAL", True))
    eval_every_steps = int(
        config.get(
            "EVAL_EVERY_STEPS",
            config.get("TEST_EVERY_STEPS", 1_000_000),
        )
    )
    eval_num_envs = int(config.get("EVAL_NUM_ENVS", config.get("TEST_NUM_ENVS", 128)))
    eval_num_steps = int(config.get("EVAL_NUM_STEPS", config.get("TEST_NUM_STEPS", 10_000)))
    eval_mods = _as_list(config.get("EVAL_MODS", config.get("MOD_NAME", ["lazy_enemy"])))
    eval_record_video = bool(config.get("EVAL_RECORD_VIDEO", config.get("RECORD_VIDEO", False)))
    eval_video_base = bool(config.get("EVAL_VIDEO_BASE", True))
    eval_video_mods = bool(config.get("EVAL_VIDEO_MODS", True))
    eval_video_max_steps = int(config.get("EVAL_VIDEO_MAX_STEPS", 1_000))
    eval_video_fps = int(config.get("EVAL_VIDEO_FPS", 30))
    accel_video_overlay_height = int(config.get("ACCEL_VIDEO_OVERLAY_HEIGHT", 24))
    train_record_accel_video = bool(config.get("TRAIN_RECORD_ACCEL_VIDEO", False))
    train_accel_video_count = max(0, int(config.get("TRAIN_ACCEL_VIDEO_COUNT", 5)))
    train_accel_video_max_steps = int(
        config.get("TRAIN_ACCEL_VIDEO_MAX_STEPS", eval_video_max_steps)
    )

    eval_env_entries = []
    if eval_during_training or final_eval:
        eval_env_entries.append(("base", apply_wrappers(jaxatari.make(config["ENV_NAME"].lower()))))
        for mod_name in eval_mods:
            mod_name = str(mod_name).lower()
            eval_env_entries.append(
                (
                    f"mod/{_safe_metric_fragment(mod_name)}",
                    apply_wrappers(jaxatari.make(config["ENV_NAME"].lower(), mods=[mod_name])),
                )
            )

    if not hasattr(env, "_mod_specs"):
        raise ValueError("ACCEL wrapper missing _mod_specs")
    mod_action_space_n = len(env._mod_specs)
    def spec_display_name(spec):
        return getattr(spec, "label", None) or spec.target

    mod_spec_metric_names = [
        f"accel_actions/spec_{i:02d}_{_safe_metric_fragment(spec_display_name(spec))}"
        for i, spec in enumerate(env._mod_specs)
    ]
    parameter_value_names = (
        "ball_x_multiplier",
        "ball_y_multiplier",
        "ball_vel_x_multiplier",
        "ball_vel_y_multiplier",
        "player_y_multiplier",
        "enemy_y_multiplier",
        "player_speed_multiplier",
        "player_paddle_height",
        "player_paddle_width",
        "enemy_step_size",
    )
    if level_encoding == "parameter_vector" and len(parameter_value_names) != mod_action_space_n:
        raise ValueError("Pong parameter-level names no longer match wrapper dimensions.")
    parameter_metric_names = [
        f"accel_parameters/{_safe_metric_fragment(name)}"
        for name in parameter_value_names[:mod_action_space_n]
    ]
    conflicting_mask_pairs = tuple(getattr(env, "_conflicting_mask_pairs", ()))
    max_level_size = min(
        int(config.get("ACCEL_MAX_LEVEL_SIZE", mod_action_space_n)),
        mod_action_space_n,
    )
    duplicate_check = bool(
        config.get("BUFFER_DUPLICATE_CHECK", config.get("buffer_duplicate_check", True))
    )
    requested_sampler_capacity = int(
        config.get("ACCEL_BUFFER_CAPACITY", config.get("LEVEL_BUFFER_CAPACITY", 64))
    )
    if level_encoding == "parameter_vector":
        parameter_bin_counts = env.parameter_bin_counts
        parameter_baseline_level = env.parameter_baseline_level
        reachable_level_count = int(np.prod(np.asarray(parameter_bin_counts)))
    else:
        reachable_level_count = sum(
            comb(mod_action_space_n, active_count)
            for active_count in range(max_level_size + 1)
        )
        # With at most two active bits, every mutually exclusive pair removes
        # exactly one otherwise-valid two-bit mask (PH+SH for Pong).
        if max_level_size <= 2:
            unique_conflicts = {
                tuple(sorted((left, right)))
                for left, right in conflicting_mask_pairs
                if left != right
            }
            reachable_level_count -= len(unique_conflicts)
    sampler_capacity = (
        min(requested_sampler_capacity, reachable_level_count)
        if duplicate_check
        else requested_sampler_capacity
    )
    if sampler_capacity != requested_sampler_capacity:
        print(
            "Clamping ACCEL_BUFFER_CAPACITY from "
            f"{requested_sampler_capacity} to {sampler_capacity}: only "
            f"{reachable_level_count} unique {level_encoding} levels are reachable."
        )

    level_sampler = LevelSampler(
        capacity=sampler_capacity,
        replay_prob=float(config.get("REPLAY_PROB", config.get("replay_prob", 0.8))),
        staleness_coeff=float(config.get("STALENESS_COEFF", config.get("staleness_coeff", 0.3))),
        minimum_fill_ratio=float(config.get("MINIMUM_FILL_RATIO", config.get("minimum_fill_ratio", 0.5))),
        prioritization=str(config.get("PRIORITIZATION", config.get("prioritization", "rank"))),
        prioritization_params={
            "temperature": float(config.get("TEMPERATURE", config.get("temperature", 0.3))),
            "k": int(config.get("TOPK_K", config.get("topk_k", 4))),
        },
        duplicate_check=duplicate_check,
    )

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES_DECAY"],
    )
    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-20,
        transition_steps=(
            config["NUM_UPDATES_DECAY"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]
        ),
    )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
    ppo_rnn_hidden_size = int(config.get("PPO_RNN_HIDDEN_SIZE", 256))

    obs_shape = tuple(env.observation_space().shape)
    if protagonist_alg == "jaxued_ppo":
        network = AtariRNNActorCritic(
            action_dim=env.action_space().n,
            obs_shape=obs_shape,
            rnn_hidden_size=ppo_rnn_hidden_size,
            hidden_size=int(config.get("PPO_HIDDEN_SIZE", 128)),
        )
    else:
        network = QNetwork(
            action_dim=env.action_space().n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            object_centric=config.get("OBJECT_CENTRIC", True),
        )

    def create_agent(rng):
        if protagonist_alg == "jaxued_ppo":
            init_obs = jnp.zeros((1, config["NUM_ENVS"], *obs_shape))
            init_dones = jnp.zeros((1, config["NUM_ENVS"]), dtype=bool)
            init_hstate = initialize_actor_critic_carry((config["NUM_ENVS"],), ppo_rnn_hidden_size)
            network_variables = network.init(rng, (init_obs, init_dones), init_hstate)
            params = network_variables["params"]
            batch_stats = {}
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr, eps=1e-5),
            )
        else:
            init_x = jnp.zeros((1, *obs_shape))
            network_variables = network.init(rng, init_x, train=False)
            params = network_variables["params"]
            batch_stats = network_variables["batch_stats"]
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
        if level_encoding == "parameter_vector":
            placeholder_level = parameter_baseline_level
            placeholder_batch = jnp.broadcast_to(
                placeholder_level, (config["NUM_ENVS"], mod_action_space_n)
            )
        else:
            placeholder_level = jnp.zeros((mod_action_space_n,), dtype=bool)
            placeholder_batch = jnp.zeros(
                (config["NUM_ENVS"], mod_action_space_n), dtype=bool
            )
        sampler = level_sampler.initialize(placeholder_level)
        return CustomTrainState.create(
            apply_fn=network.apply,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            sampler=sampler,
            update_state=jnp.array(UpdateState.DR, dtype=jnp.int32),
            last_replay_level_batch=placeholder_batch,
            dr_last_level_batch=placeholder_batch,
            mutation_last_level_batch=placeholder_batch,
            current_level_batch=placeholder_batch,
        )

    def level_round_metrics(levels, strength):
        """Log decoded parameter values or legacy mask occupancy per update."""
        if level_encoding == "parameter_vector":
            values = env.decode_parameter_levels(levels)
            return {
                **{
                    parameter_metric_names[i]: jnp.mean(values[:, i])
                    for i in range(mod_action_space_n)
                },
                "accel_parameters/mean_distance_from_baseline_bins": jnp.mean(
                    jnp.abs(levels - parameter_baseline_level)
                ),
            }
        return {
            "mask/active_bits_mean": jnp.mean(jnp.sum(levels, axis=-1)),
            "mask/active_fraction": jnp.mean(levels),
            "accel_actions/level_size_mean": jnp.mean(jnp.sum(levels, axis=-1)),
            "accel_actions/level_size_max": jnp.max(jnp.sum(levels, axis=-1)),
            "accel_actions/strength": strength,
            **{
                mod_spec_metric_names[i]: jnp.mean(levels[:, i].astype(jnp.float32))
                for i in range(mod_action_space_n)
            },
        }

    def train_round_pqn(train_state, init_obs, init_env_state, rng, level_masks, strength, update_grad: bool):
        def _step_env(carry, step_idx):
            last_obs, env_state, rng = carry
            rng, rng_a = jax.random.split(rng)
            q_vals = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                last_obs,
                train=False,
            )
            eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
            action = jax.vmap(eps_greedy_exploration)(
                jax.random.split(rng_a, config["NUM_ENVS"]),
                q_vals,
                eps,
            )
            new_obs, new_env_state, reward, terminated, truncated, info = jax.vmap(
                env.step,
                in_axes=(0, 0, 0, None),
            )(
                env_state,
                action,
                level_masks,
                strength,
            )
            done = jnp.logical_or(terminated, truncated)
            transition = Transition(
                obs=last_obs,
                action=action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=done,
                next_obs=new_obs,
                q_val=q_vals,
            )
            return (new_obs, new_env_state, rng), (transition, info)

        step_indices = jnp.arange(config["NUM_STEPS"], dtype=jnp.int32)
        (last_obs, final_env_state, rng), (transitions, infos) = jax.lax.scan(
            _step_env,
            (init_obs, init_env_state, rng),
            step_indices,
        )

        train_state = train_state.replace(
            timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
        )

        last_q = network.apply(
            {"params": train_state.params, "batch_stats": train_state.batch_stats},
            transitions.next_obs[-1],
            train=False,
        )
        last_q = jnp.max(last_q, axis=-1)

        def _get_target(lambda_returns_and_next_q, transition):
            lambda_returns, next_q = lambda_returns_and_next_q
            target_bootstrap = (
                transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
            )
            delta = lambda_returns - next_q
            lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
            lambda_returns = (
                (1 - transition.done) * lambda_returns
                + transition.done * transition.reward
            )
            next_q = jnp.max(transition.q_val, axis=-1)
            return (lambda_returns, next_q), lambda_returns

        last_q = last_q * (1 - transitions.done[-1])
        lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
        _, targets = jax.lax.scan(
            _get_target,
            (lambda_returns, last_q),
            jax.tree_util.tree_map(lambda x: x[:-1], transitions),
            reverse=True,
        )
        lambda_targets = jnp.concatenate((targets, lambda_returns[None]))

        chosen_q = jnp.take_along_axis(
            transitions.q_val,
            transitions.action[..., None],
            axis=-1,
        ).squeeze(-1)
        greedy_actions = jnp.argmax(transitions.q_val, axis=-1)
        scores = _compute_mask_scores(
            config,
            transitions.done,
            transitions.reward,
            chosen_q,
            lambda_targets,
        )

        def _learn(train_state_and_rng):
            train_state, rng = train_state_and_rng

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):
                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)
                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn,
                        has_aux=True,
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(rng, x)
                    x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng),
                    transitions,
                )
                targets_mb = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng),
                    lambda_targets,
                )

                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase,
                    (train_state, rng),
                    (minibatches, targets_mb),
                )
                return (train_state, rng), (loss, qvals)

            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch,
                (train_state, rng),
                None,
                config["NUM_EPOCHS"],
            )
            return train_state, rng, loss.mean(), qvals.mean()

        def _skip_learn(train_state_and_rng):
            train_state, rng = train_state_and_rng
            return train_state, rng, jnp.array(0.0, dtype=jnp.float32), chosen_q.mean()

        train_state, rng, loss_mean, qvals_mean = jax.lax.cond(
            update_grad,
            _learn,
            _skip_learn,
            (train_state, rng),
        )
        train_state = train_state.replace(n_updates=train_state.n_updates + 1)

        round_metrics = {
            "grad_steps": train_state.grad_steps,
            # New/mutated ACCEL levels are intentionally score-only when
            # EXPLORATORY_GRAD_UPDATES is false.  Do not present their
            # placeholder zero as a training loss in W&B: it makes a healthy
            # replay-only update schedule look like a constant, stalled loss.
            "td_loss": jnp.where(update_grad, loss_mean, jnp.nan),
            "agent/gradient_applied": jnp.asarray(update_grad, dtype=jnp.int32),
            "agent/epsilon": eps_scheduler(jnp.maximum(train_state.n_updates - 1, 0)),
            "agent/greedy_non_noop_fraction": jnp.mean(greedy_actions != 0),
            "agent/selected_non_noop_fraction": jnp.mean(transitions.action != 0),
            "agent/nonzero_reward_fraction": jnp.mean(transitions.reward != 0),
            "qvals": qvals_mean,
            "level_sampler/score_mean": jnp.mean(scores),
            **_mean_infos(
                infos,
                skip=("level_mask", "level_parameters", "parameter_values", "strength"),
            ),
        }
        round_metrics.update(level_round_metrics(level_masks, strength))
        round_metrics.update(
            {
                f"agent/greedy_action_{action_index}": jnp.mean(
                    greedy_actions == action_index
                )
                for action_index in range(env.action_space().n)
            }
        )
        return train_state, rng, scores, round_metrics

    def train_round_ppo(train_state, init_obs, init_env_state, rng, level_masks, strength, update_grad: bool):
        init_hstate = initialize_actor_critic_carry((config["NUM_ENVS"],), ppo_rnn_hidden_size)

        def _step_env(carry, _):
            rng, hstate, obs, env_state, last_done = carry
            rng, rng_action = jax.random.split(rng)

            x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
            hstate, pi, value = train_state.apply_fn({"params": train_state.params}, x, hstate)
            action = pi.sample(seed=rng_action)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            next_obs, next_env_state, reward, terminated, truncated, info = jax.vmap(
                env.step,
                in_axes=(0, 0, 0, None),
            )(
                env_state,
                action,
                level_masks,
                strength,
            )
            done = jnp.logical_or(terminated, truncated)
            carry = (rng, hstate, next_obs, next_env_state, done)
            return carry, (obs, action, reward, done, log_prob, value, info)

        (
            rng,
            final_hstate,
            last_obs,
            final_env_state,
            last_done,
        ), traj = jax.lax.scan(
            _step_env,
            (
                rng,
                init_hstate,
                init_obs,
                init_env_state,
                jnp.zeros(config["NUM_ENVS"], dtype=bool),
            ),
            None,
            length=config["NUM_STEPS"],
        )
        obs, actions, rewards, dones, log_probs, values, infos = traj

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
        _, _, last_value = train_state.apply_fn({"params": train_state.params}, x, final_hstate)
        last_value = last_value.squeeze(0)

        advantages, targets = compute_gae(
            config["GAMMA"],
            config["LAMBDA"],
            last_value,
            values,
            rewards,
            dones,
        )

        score_fn = str(config.get("JAXUED_SCORE_FUNCTION", "pvl")).lower()
        incomplete_value = float(config.get("JAXUED_INCOMPLETE_SCORE", 0.0))
        if score_fn in {"max_mc", "maxmc"}:
            scores = max_mc(dones, values, compute_max_returns(dones, rewards), incomplete_value=incomplete_value)
        else:
            scores = positive_value_loss(dones, advantages, incomplete_value=incomplete_value)

        (rng, train_state), losses = update_actor_critic_rnn(
            rng,
            train_state,
            init_hstate,
            (obs, actions, dones, log_probs, values, targets, advantages),
            config["NUM_ENVS"],
            config["NUM_STEPS"],
            config["NUM_MINIBATCHES"],
            config["NUM_EPOCHS"],
            float(config.get("PPO_CLIP_EPS", config.get("clip_eps", 0.2))),
            float(config.get("PPO_ENTROPY_COEFF", config.get("entropy_coeff", 1e-3))),
            float(config.get("PPO_CRITIC_COEFF", config.get("critic_coeff", 0.5))),
            update_grad=update_grad,
        )
        train_state = train_state.replace(
            timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"],
            n_updates=train_state.n_updates + 1,
        )

        loss_mean = jax.tree_util.tree_map(lambda x: x.mean(), losses)
        round_metrics = {
            "grad_steps": train_state.grad_steps,
            "ppo/loss": jnp.where(update_grad, loss_mean[0], jnp.nan),
            "agent/gradient_applied": jnp.asarray(update_grad, dtype=jnp.int32),
            "level_sampler/score_mean": jnp.mean(scores),
            "return/rollout_mean": jnp.mean(rewards),
            **_mean_infos(
                infos,
                skip=("level_mask", "level_parameters", "parameter_values", "strength"),
            ),
        }
        round_metrics.update(level_round_metrics(level_masks, strength))
        return train_state, rng, scores, round_metrics

    train_round = train_round_ppo if protagonist_alg == "jaxued_ppo" else train_round_pqn

    def record_rollout_video(
        train_state,
        rollout_env,
        label: str,
        env_step: int,
        rng,
        level_mask=None,
        strength: float = 1.0,
        max_steps: int | None = None,
    ):
        """Render one greedy rollout and send it to W&B from the host.

        Evaluation environments use their ordinary ``step`` signature.  Passing
        ``level_mask`` selects the ACCEL signature and records exactly the
        mask or parameter vector used by the most recent curriculum update.
        """
        max_steps = eval_video_max_steps if max_steps is None else max_steps
        if max_steps <= 0:
            return

        obs, env_state = rollout_env.reset(rng)
        renderer = rollout_env.renderer
        frames = []
        total_reward = 0.0
        last_done = jnp.asarray(False)
        hstate = (
            initialize_actor_critic_carry((1,), ppo_rnn_hidden_size)
            if protagonist_alg == "jaxued_ppo"
            else None
        )

        for _ in range(max_steps):
            if protagonist_alg == "jaxued_ppo":
                policy_input = (
                    obs[None, None, ...],
                    last_done[None, None],
                )
                hstate, pi, _ = train_state.apply_fn(
                    {"params": train_state.params}, policy_input, hstate
                )
                action = pi.mode().reshape(())
            else:
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    obs[None, ...],
                    train=False,
                )
                action = jnp.argmax(q_vals, axis=-1)[0]

            if level_mask is None:
                obs, env_state, reward, terminated, truncated, _ = rollout_env.step(
                    env_state, action
                )
            else:
                obs, env_state, reward, terminated, truncated, _ = rollout_env.step(
                    env_state, action, level_mask, strength
                )
            last_done = jnp.logical_or(terminated, truncated)
            frame = renderer.render(_unwrap_state_for_render(env_state))
            frames.append(np.asarray(jax.device_get(frame), dtype=np.uint8))
            total_reward += float(jax.device_get(reward))
            if bool(jax.device_get(last_done)):
                break

        if not frames:
            return
        frames = np.stack(frames, axis=0)

        # The top bar identifies the dimensions that differ from base Pong.
        # In parameter-vector mode the W&B caption and scalar metrics contain
        # exact decoded values; the highlighted panels show which dimensions
        # are currently non-baseline directly in the video pixels.
        metrics_payload = {
            "num_env_steps": env_step,
            f"video_metrics/{label}/return": total_reward,
            f"video_metrics/{label}/length": len(frames),
        }
        if level_mask is not None:
            is_parameter_level = (
                getattr(rollout_env, "_level_encoding", "mask") == "parameter_vector"
            )
            raw_level = np.asarray(jax.device_get(level_mask))
            if is_parameter_level:
                mask = raw_level != np.asarray(
                    jax.device_get(rollout_env.parameter_baseline_level)
                )
            else:
                mask = raw_level.astype(bool)
            action_targets = [spec_display_name(spec) for spec in rollout_env._mod_specs]
            action_labels = [_short_accel_label(target) for target in action_targets]
            _overlay_accel_mask_banner(
                frames,
                mask,
                action_labels,
                height=accel_video_overlay_height,
            )
            if is_parameter_level:
                decoded = np.asarray(
                    jax.device_get(rollout_env.decode_parameter_level(level_mask))
                )
                display_names = parameter_value_names[: len(decoded)]
                for index, value in enumerate(decoded):
                    metric_name = _safe_metric_fragment(display_names[index])
                    metrics_payload[f"video_metrics/{label}/parameter/{metric_name}"] = float(value)
                metrics_payload[f"video_metrics/{label}/changed_dimensions"] = int(mask.sum())
                video_caption = (
                    "ACCEL parameters: "
                    + ", ".join(
                        f"{name}={value:g}" for name, value in zip(display_names, decoded)
                    )
                    + " | highlighted: dimensions differing from base Pong"
                )
            else:
                for index, active in enumerate(mask):
                    metric_name = mod_spec_metric_names[index].split("/")[-1]
                    metrics_payload[f"video_metrics/{label}/mask/{metric_name}"] = float(active)
                metrics_payload[f"video_metrics/{label}/active_bits"] = int(mask.sum())
                metrics_payload[f"video_metrics/{label}/strength"] = float(strength)
                active_targets = [
                    target for target, active in zip(action_targets, mask) if active
                ]
                video_caption = (
                    f"ACCEL active: {', '.join(active_targets) if active_targets else 'none'} "
                    f"| mask: {''.join('1' if active else '0' for active in mask)} "
                    f"| legend: {', '.join(f'{short}={target}' for short, target in zip(action_labels, action_targets))}"
                )
        else:
            video_caption = "Evaluation rollout without an ACCEL mask."
        video_payload = {
            "num_env_steps": env_step,
            f"video/{label}": wandb.Video(
                np.transpose(frames, (0, 3, 1, 2)),
                fps=eval_video_fps,
                format="mp4",
                caption=video_caption,
            ),
        }
        wandb.log(video_payload, step=env_step)
        wandb.log(metrics_payload, step=env_step)

    def _build_eval_fn(eval_env):
        @jax.jit
        def _eval(train_state, rng):
            def _step_pqn(carry, _):
                env_state, obs, rng = carry
                q_vals = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    obs,
                    train=False,
                )
                action = jnp.argmax(q_vals, axis=-1)
                next_obs, next_env_state, reward, terminated, truncated, info = jax.vmap(eval_env.step)(
                    env_state,
                    action,
                )
                return (next_env_state, next_obs, rng), info

            def _step_ppo(carry, _):
                env_state, obs, rng, hstate, last_done = carry
                x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
                hstate, pi, value = train_state.apply_fn({"params": train_state.params}, x, hstate)
                action = pi.mode().squeeze(0)
                next_obs, next_env_state, reward, terminated, truncated, info = jax.vmap(eval_env.step)(
                    env_state,
                    action,
                )
                done = jnp.logical_or(terminated, truncated)
                return (next_env_state, next_obs, rng, hstate, done), info

            rng, rng_reset = jax.random.split(rng)
            init_obs, env_state = jax.vmap(eval_env.reset)(jax.random.split(rng_reset, eval_num_envs))
            if protagonist_alg == "jaxued_ppo":
                init_hstate = initialize_actor_critic_carry((eval_num_envs,), ppo_rnn_hidden_size)
                _, infos = jax.lax.scan(
                    _step_ppo,
                    (
                        env_state,
                        init_obs,
                        rng,
                        init_hstate,
                        jnp.zeros(eval_num_envs, dtype=bool),
                    ),
                    None,
                    eval_num_steps,
                )
            else:
                _, infos = jax.lax.scan(
                    _step_pqn,
                    (env_state, init_obs, rng),
                    None,
                    eval_num_steps,
                )

            returned = infos["returned_episode"]
            returned_f = returned.astype(jnp.float32)
            denom = jnp.maximum(jnp.sum(returned_f), 1.0)

            def _masked_mean(x):
                x = x.squeeze()
                if x.dtype == jnp.bool_:
                    x = x.astype(jnp.float32)
                mask = jnp.broadcast_to(returned, x.shape)
                return jnp.sum(jnp.where(mask, x, 0.0)) / denom

            done_infos = jax.tree_util.tree_map(_masked_mean, infos)
            return {
                "returned_episode_returns": done_infos["returned_episode_returns"],
                "returned_episode_lengths": done_infos["returned_episode_lengths"],
                "episodes": jnp.sum(returned_f),
            }

        return _eval

    eval_fns = [
        (label, _build_eval_fn(eval_env), eval_env)
        for label, eval_env in eval_env_entries
    ]

    @jax.jit
    def train_step(carry, _):
        rng, train_state = carry
        strength = jnp.asarray(config.get("ACCEL_STRENGTH", 1.0), dtype=jnp.float32)

        def reset_batch(rng):
            return jax.vmap(env.reset)(jax.random.split(rng, config["NUM_ENVS"]))

        def random_level_batch(rng):
            if level_encoding == "parameter_vector":
                baseline_prob = float(
                    config.get("ACCEL_PARAMETER_BASELINE_PROB", 0.15)
                )
                return jax.vmap(
                    lambda key: _make_random_parameter_level(
                        key,
                        parameter_bin_counts,
                        parameter_baseline_level,
                        baseline_prob,
                    )
                )(jax.random.split(rng, config["NUM_ENVS"]))
            return jax.vmap(
                lambda key: _make_random_mask(
                    key,
                    mod_action_space_n,
                    max_level_size,
                    conflicting_mask_pairs,
                )
            )(jax.random.split(rng, config["NUM_ENVS"]))

        def on_new_levels(rng, train_state):
            rng, rng_levels, rng_reset, rng_rollout = jax.random.split(rng, 4)
            levels = random_level_batch(rng_levels)
            init_obs, init_env_state = reset_batch(rng_reset)
            train_state, rng_rollout, scores, round_metrics = train_round(
                train_state,
                init_obs,
                init_env_state,
                rng_rollout,
                levels,
                strength,
                bool(config.get("EXPLORATORY_GRAD_UPDATES", True)),
            )
            sampler, _ = level_sampler.insert_batch(train_state.sampler, levels, scores)
            train_state = train_state.replace(
                sampler=sampler,
                update_state=jnp.array(UpdateState.DR, dtype=jnp.int32),
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=levels,
                current_level_batch=levels,
            )
            return (rng_rollout, train_state), round_metrics

        def on_replay_levels(rng, train_state):
            rng, rng_levels, rng_reset, rng_rollout = jax.random.split(rng, 4)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                train_state.sampler,
                rng_levels,
                config["NUM_ENVS"],
            )
            init_obs, init_env_state = reset_batch(rng_reset)
            train_state = train_state.replace(sampler=sampler)
            train_state, rng_rollout, scores, round_metrics = train_round(
                train_state,
                init_obs,
                init_env_state,
                rng_rollout,
                levels,
                strength,
                True,
            )
            sampler = level_sampler.update_batch(train_state.sampler, level_inds, scores)
            train_state = train_state.replace(
                sampler=sampler,
                update_state=jnp.array(UpdateState.REPLAY, dtype=jnp.int32),
                num_replay_updates=train_state.num_replay_updates + 1,
                last_replay_level_batch=levels,
                current_level_batch=levels,
            )
            return (rng_rollout, train_state), round_metrics

        def on_mutate_levels(rng, train_state):
            rng, rng_mutate, rng_reset, rng_rollout = jax.random.split(rng, 4)
            parent_levels = jax.lax.cond(
                train_state.num_replay_updates > 0,
                lambda: train_state.last_replay_level_batch,
                lambda: (
                    jnp.broadcast_to(
                        parameter_baseline_level,
                        (config["NUM_ENVS"], mod_action_space_n),
                    )
                    if level_encoding == "parameter_vector"
                    else jnp.zeros((config["NUM_ENVS"], mod_action_space_n), dtype=bool)
                ),
            )
            if level_encoding == "parameter_vector":
                levels = jax.vmap(
                    lambda key, level: _mutate_parameter_level(
                        key,
                        level,
                        parameter_bin_counts,
                        float(config.get("ACCEL_MUTATION_MULTI_PARAM_PROB", 0.15)),
                    )
                )(
                    jax.random.split(rng_mutate, config["NUM_ENVS"]), parent_levels
                )
            else:
                levels = jax.vmap(
                    lambda key, level: _mutate_mask(
                        key,
                        level,
                        max_level_size,
                        conflicting_mask_pairs,
                    )
                )(
                    jax.random.split(rng_mutate, config["NUM_ENVS"]),
                    parent_levels,
                )
            init_obs, init_env_state = reset_batch(rng_reset)
            train_state, rng_rollout, scores, round_metrics = train_round(
                train_state,
                init_obs,
                init_env_state,
                rng_rollout,
                levels,
                strength,
                bool(config.get("EXPLORATORY_GRAD_UPDATES", True)),
            )
            sampler, _ = level_sampler.insert_batch(train_state.sampler, levels, scores)
            train_state = train_state.replace(
                sampler=sampler,
                update_state=jnp.array(UpdateState.DR, dtype=jnp.int32),
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=levels,
                current_level_batch=levels,
            )
            return (rng_rollout, train_state), round_metrics

        rng, rng_replay = jax.random.split(rng)
        replay_decision = (
            level_sampler.sample_replay_decision(train_state.sampler, rng_replay)
            & (train_state.sampler["size"] > 0)
        ).astype(jnp.int32)
        if alg_mode == "accel":
            branch = (1 - train_state.update_state) * replay_decision + 2 * train_state.update_state
        else:
            branch = replay_decision

        (rng, train_state), metrics = jax.lax.switch(
            branch,
            [on_new_levels, on_replay_levels, on_mutate_levels],
            rng,
            train_state,
        )
        metrics = {
            **metrics,
            "updates/dr": train_state.num_dr_updates,
            "updates/replay": train_state.num_replay_updates,
            "updates/mutation": train_state.num_mutation_updates,
            "branch": branch,
        }
        return (rng, train_state), metrics

    def train(rng):
        rng, agent_rng = jax.random.split(rng)
        train_state = create_agent(agent_rng)

        metrics_history = []
        runner_state = (rng, train_state)
        prev_time = time.time()
        next_eval_step = eval_every_steps
        evaluation_index = 0
        total_train_steps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
        train_video_targets = (
            np.ceil(
                np.linspace(
                    total_train_steps / train_accel_video_count,
                    total_train_steps,
                    train_accel_video_count,
                )
            ).astype(int)
            if train_record_accel_video and train_accel_video_count > 0
            else np.empty(0, dtype=int)
        )
        train_video_index = 0

        def evaluate_snapshot(train_state, rng, env_step, metric_prefix, video_prefix):
            """Run all configured eval envs and write uniquely named videos."""
            snapshot_metrics = {}
            for label, eval_fn, eval_env in eval_fns:
                rng, eval_rng = jax.random.split(rng)
                eval_metrics = eval_fn(train_state, eval_rng)
                safe_label = _safe_metric_fragment(label)
                snapshot_metrics.update(
                    {
                        f"{metric_prefix}/{safe_label}/returned_episode_returns": float(
                            jax.device_get(eval_metrics["returned_episode_returns"])
                        ),
                        f"{metric_prefix}/{safe_label}/returned_episode_lengths": float(
                            jax.device_get(eval_metrics["returned_episode_lengths"])
                        ),
                        f"{metric_prefix}/{safe_label}/episodes": float(
                            jax.device_get(eval_metrics["episodes"])
                        ),
                    }
                )
                record_this_eval_video = (
                    (label == "base" and eval_video_base)
                    or (label != "base" and eval_video_mods)
                )
                if (
                    eval_record_video
                    and record_this_eval_video
                    and config["WANDB_MODE"] != "disabled"
                ):
                    rng, video_rng = jax.random.split(rng)
                    # The evaluation index is part of the key, preventing W&B
                    # from replacing every previous base/mod clip in-place.
                    record_rollout_video(
                        train_state,
                        eval_env,
                        f"{video_prefix}/{safe_label}",
                        env_step,
                        video_rng,
                    )
            return rng, snapshot_metrics

        for _ in range(config["NUM_UPDATES"]):
            runner_state, metrics = train_step(runner_state, None)
            rng = runner_state[0]
            train_state = runner_state[1]
            now = time.time()
            elapsed = max(now - prev_time, 1e-8)
            prev_time = now
            sampler_metrics = train_state_to_log_dict(train_state, level_sampler)
            env_steps_per_update = config["NUM_STEPS"] * config["NUM_ENVS"]
            update_count = (
                train_state.num_dr_updates
                + train_state.num_replay_updates
                + train_state.num_mutation_updates
            )
            metrics_host = {
                "env_step": int(jax.device_get(train_state.timesteps)),
                "update_steps": int(jax.device_get(train_state.n_updates)),
                "num_updates": int(jax.device_get(update_count)),
                "num_env_steps": int(jax.device_get(train_state.timesteps)),
                "sps": float(env_steps_per_update / elapsed),
                **_host_metrics(metrics),
                **_host_metrics(sampler_metrics),
            }
            if eval_during_training and metrics_host["env_step"] >= next_eval_step:
                evaluation_index += 1
                rng, periodic_eval_metrics = evaluate_snapshot(
                    train_state,
                    rng,
                    metrics_host["env_step"],
                    "eval",
                    f"eval_{evaluation_index:03d}",
                )
                metrics_host.update(periodic_eval_metrics)
                while next_eval_step <= metrics_host["env_step"]:
                    next_eval_step += eval_every_steps
            while (
                train_video_index < len(train_video_targets)
                and metrics_host["env_step"] >= train_video_targets[train_video_index]
            ):
                if config["WANDB_MODE"] != "disabled":
                    rng, video_rng = jax.random.split(rng)
                    record_rollout_video(
                        train_state,
                        env,
                        f"train/accel_{train_video_index + 1:02d}",
                        metrics_host["env_step"],
                        video_rng,
                        level_mask=train_state.current_level_batch[0],
                        strength=float(config.get("ACCEL_STRENGTH", 1.0)),
                        max_steps=train_accel_video_max_steps,
                    )
                train_video_index += 1
            runner_state = (rng, train_state)
            metrics_history.append(metrics_host)
            if config["WANDB_MODE"] != "disabled":
                wandb.log(metrics_host, step=metrics_host["env_step"])

        if final_eval:
            final_env_step = int(jax.device_get(train_state.timesteps))
            rng, final_eval_metrics = evaluate_snapshot(
                train_state,
                rng,
                final_env_step,
                "eval/final",
                "final",
            )
            final_metrics_host = {
                "env_step": final_env_step,
                "num_env_steps": final_env_step,
                **final_eval_metrics,
            }
            metrics_history.append(final_metrics_host)
            if config["WANDB_MODE"] != "disabled":
                wandb.log(final_metrics_host, step=final_env_step)

        runner_state = (rng, train_state)
        return {"runner_state": runner_state, "metrics": metrics_history}

    return train


def single_run(config):
    if "alg" in config:
        config = {**config, **config["alg"]}
    env_name = config["ENV_NAME"]
    oc = "oc" if config.get("OBJECT_CENTRIC", False) else "pixel"
    alg_name = config.get("ALG_NAME", f"pqn_jaxued_{config.get('JAXUED_ALG', 'accel')}")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), "JAXUED_LEVELSAMPLER", f"jax_{jax.__version__}"],
        name=config.get("NAME", f"{alg_name}_{env_name}"),
        config=config,
        mode=config["WANDB_MODE"],
    )
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval/*", step_metric="num_env_steps")
    wandb.define_metric("video/*", step_metric="num_env_steps")
    wandb.define_metric("video_metrics/*", step_metric="num_env_steps")
    wandb.define_metric("mask/*", step_metric="num_updates")
    wandb.define_metric("morphology/*", step_metric="num_updates")
    wandb.define_metric("accel_actions/*", step_metric="num_updates")
    wandb.define_metric("accel_parameters/*", step_metric="num_updates")
    wandb.define_metric("updates/*", step_metric="num_updates")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    t0 = time.time()
    outs = []
    for seed_idx, seed_rng in enumerate(rngs):
        seed_config = copy.deepcopy(config)
        seed_config["SEED"] = int(config["SEED"]) + seed_idx
        out = make_train_jaxatari_accel_plr(seed_config)(seed_rng)
        jax.block_until_ready(out["runner_state"][1].timesteps)
        outs.append(out)
    print(f"Total: {time.time() - t0} seconds.")

    if config.get("SAVE_PATH", None) is not None and outs:
        model_state = outs[0]["runner_state"][1]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f"{alg_name}_{env_name}_{oc}_seed{config['SEED']}_config.yaml"),
        )
        save_path = os.path.join(
            save_dir,
            f"{alg_name}_{env_name}_{oc}_seed{config['SEED']}_vmap0.safetensors",
        )
        save_params(model_state.params, save_path)
        save_params(model_state.batch_stats, save_path.replace(".safetensors", "_bs.safetensors"))
        print(f"Model saved to {save_dir}")

    wandb.finish()


def tune(default_config):
    raise NotImplementedError("Add a sweep loop after the JAXUED LevelSampler runner is validated.")


@hydra.main(version_base=None, config_path="config/alg")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
