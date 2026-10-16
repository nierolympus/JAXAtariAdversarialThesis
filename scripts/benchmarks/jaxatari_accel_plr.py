"""
PQN + JAXUED-style PLR/ACCEL for JAXAtari adversarial levels.

This runner keeps the JAXAtari wrapper workflow from ACCEL.py, but replaces
the host-side ACCELLevelBuffer with JAXUED's LevelSampler and the scheduling
logic used by jaxued/examples/maze_plr.py.

Level meaning in this file:
    Maze PLR level       -> full maze layout
    JAXAtari level here  -> a bool adversarial mask, a discrete parameter
                            vector, or a game-provided structured layout PyTree

ACCEL mode follows Maze PLR's branch schedule:
    0: new random levels
    1: replay sampled levels from LevelSampler
    2: mutate the last replayed levels and insert/update them
    3: repeat the last replayed levels before considering a mutation
"""

from __future__ import annotations

import copy
import os
import re
import sys
import time
from enum import IntEnum
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
from jaxatari.accel_wrappers import (
    FreewayAccelModWrapper,
    KangarooAccelWrapper,
    PongAccelModWrapper,
    SeaquestAccelModWrapper,
)
from jaxatari.wrappers import (
    AtariWrapper,
    FlattenObservationWrapper,
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
    last_replay_level_inds: Any
    dr_last_level_batch: Any
    mutation_last_level_batch: Any
    current_level_batch: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    num_dr_updates: int = 0
    num_replay_updates: int = 0
    num_mutation_updates: int = 0


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


def compute_dense_q_pvl(values, lambda_targets):
    """Return a truncated positive-value-loss regret proxy per environment.

    ``lambda_targets - values`` is the Q-learning analogue of the GAE term
    used by PLR's positive value loss. Unlike JAXUED's episodic helper, this
    averages every transition, so a long Pong match need not finish inside a
    short rollout. Auto-reset transitions remain valid because one training
    round keeps the same level.
    """
    positive_value_error = jnp.maximum(
        jax.lax.stop_gradient(lambda_targets - values),
        0.0,
    )
    return jnp.mean(positive_value_error, axis=0)


def compute_per_env_regret(chosen_q, lambda_targets, dones=None):
    """Backward-compatible alias for the former dense Q-PVL helper."""
    del dones
    return compute_dense_q_pvl(chosen_q, lambda_targets)


def compute_q_lambda_targets(rewards, dones, q_values, last_q, gamma, lambda_):
    """Compute correctly aligned truncated Q(lambda) targets.

    ``q_values[t + 1]`` supplies Q(s_{t+1}) for transition ``t``; only the
    final transition bootstraps from the separately evaluated ``last_q``.
    """
    next_q_values = jnp.concatenate(
        (
            jnp.max(q_values[1:], axis=-1),
            last_q[None, :],
        ),
        axis=0,
    )

    def _get_target(next_lambda_return, transition_and_next_q):
        reward, done, next_q = transition_and_next_q
        not_done = 1.0 - done.astype(jnp.float32)
        lambda_return = reward + gamma * not_done * (
            (1.0 - lambda_) * next_q + lambda_ * next_lambda_return
        )
        return lambda_return, lambda_return

    _, lambda_targets = jax.lax.scan(
        _get_target,
        last_q,
        (rewards, dones, next_q_values),
        reverse=True,
    )
    return lambda_targets


def compute_env_returns(env_rewards):
    """Sum raw environment rewards over the rollout time axis."""
    return jnp.sum(
        jnp.asarray(env_rewards, dtype=jnp.float32),
        axis=0,
    )


def compute_env_return_score(env_rewards, higher_is_better: bool = True):
    """Turn raw environment rewards into a generic level-difficulty score.

    JAXUED's sampler prioritizes larger scores. Most reinforcement-learning
    environments define larger returns as better for the protagonist, so the
    default score is the negative undiscounted rollout return. Environments
    that expose costs instead can set
    ``JAXUED_ENV_REWARD_HIGHER_IS_BETTER=false`` without adding a new scorer.
    """
    env_returns = compute_env_returns(env_rewards)
    return -env_returns if higher_is_better else env_returns


def compute_reward_learning_progress_score(
    env_returns,
    previous_return_ema,
    return_initialized,
    *,
    ema_alpha: float = 0.2,
    new_level_score: float = 1.0,
    epsilon: float = 1e-6,
):
    """Score reward-based learning progress and update the return EMA.

    The normalized absolute change is direction agnostic: both improvement
    and forgetting indicate that a level is currently informative. A level
    whose return remains permanently poor or permanently solved converges to
    zero priority. The normalization keeps the score in ``[0, 1]`` and makes
    it portable across environments with different reward scales.
    """
    env_returns = jnp.asarray(env_returns, dtype=jnp.float32)
    previous_return_ema = jnp.asarray(previous_return_ema, dtype=jnp.float32)
    return_initialized = jnp.asarray(return_initialized, dtype=bool)

    delta = jnp.abs(env_returns - previous_return_ema)
    scale = jnp.maximum(
        jnp.abs(env_returns) + jnp.abs(previous_return_ema),
        jnp.asarray(epsilon, dtype=jnp.float32),
    )
    progress = delta / scale
    scores = jnp.where(
        return_initialized,
        progress,
        jnp.asarray(new_level_score, dtype=jnp.float32),
    )
    updated_return_ema = jnp.where(
        return_initialized,
        previous_return_ema
        + jnp.asarray(ema_alpha, dtype=jnp.float32)
        * (env_returns - previous_return_ema),
        env_returns,
    )
    return scores, updated_return_ema


def _mean_infos(infos, skip=()):
    """Average rollout infos exactly like ``pqn_agent_adv_random.py``."""
    return {
        key: jnp.mean(value)
        for key, value in infos.items()
        if key not in skip
    }


def _completed_episode_metrics(infos):
    """Aggregate ``LogWrapper`` returns only where an episode completed.

    ``LogWrapper`` deliberately keeps its last completed return and length in
    the info dictionary between terminal steps.  A plain rollout mean would
    therefore re-log that stale value for every later transition.  Masking by
    ``returned_episode`` preserves the wrapper's useful state while making a
    training metric mean exactly "completed episodes in this update".
    """
    returned = jnp.asarray(infos["returned_episode"], dtype=bool)
    returned_f = returned.astype(jnp.float32)
    episode_count = jnp.sum(returned_f)
    denom = jnp.maximum(episode_count, 1.0)

    def _completed_mean(key):
        values = jnp.asarray(infos[key], dtype=jnp.float32)
        mask = jnp.broadcast_to(returned, values.shape)
        mean = jnp.sum(jnp.where(mask, values, 0.0)) / denom
        return jnp.where(
            episode_count > 0,
            mean,
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        )

    episode_return = _completed_mean("returned_episode_returns")
    episode_length = _completed_mean("returned_episode_lengths")
    return {
        # Preserve the wrapper's existing public names, now with correct
        # completion-only semantics for the training dashboard.
        "returned_episode_returns": episode_return,
        "returned_episode_lengths": episode_length,
        "returned_episode_count": episode_count,
    }


def _host_metrics(metrics):
    return {key: float(jax.device_get(value)) for key, value in metrics.items()}


def _safe_metric_fragment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


def _overlay_accel_mask_banner(
    frames: np.ndarray,
    mask: np.ndarray,
    labels: Sequence[str],
    height: int = 24,
) -> None:
    """Burn a readable active/inactive ACCEL-mask legend into video frames.

    The overlay occupies a narrow strip at the top of each frame. Green panels
    are active, charcoal panels are inactive, and each panel has a
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
    # native 160-pixel Atari width.
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


def _overlay_accel_bucket_banner(
    frames: np.ndarray,
    bucket_indices: np.ndarray,
    bucket_counts: np.ndarray,
    baseline_indices: np.ndarray,
    labels: Sequence[str],
    height: int = 32,
) -> None:
    """Show every ACCEL parameter value relative to its base-game setting.

    Each labelled section has a fixed white centre line for the base-game
    value. A lower selected value is blue and appears to its left; a higher
    selected value is warm-coloured and appears to its right. The coloured
    position is normalised by the available bucket distance on that side of
    the parameter's base value. This keeps the white reference in the centre
    of every section while still showing the signed size of each curriculum
    change at a glance.
    """
    if not (
        len(bucket_indices)
        == len(bucket_counts)
        == len(baseline_indices)
        == len(labels)
    ):
        raise ValueError("ACCEL bucket data and action-label lengths must match.")
    if not len(labels) or height <= 0:
        return

    banner_height = min(max(int(height), 16), frames.shape[1])
    edges = np.linspace(0, frames.shape[2], len(labels) + 1, dtype=int)
    panel_color = np.array([23, 30, 40], dtype=np.uint8)
    rail_color = np.array([74, 86, 103], dtype=np.uint8)
    baseline_tick_color = np.array([244, 244, 244], dtype=np.uint8)
    lower_near_color = np.array([81, 177, 242], dtype=np.float32)
    lower_far_color = np.array([42, 101, 205], dtype=np.float32)
    higher_near_color = np.array([255, 194, 79], dtype=np.float32)
    higher_far_color = np.array([236, 85, 55], dtype=np.float32)
    separator_color = np.array([163, 174, 190], dtype=np.uint8)

    glyphs = {
        "B": ("110", "101", "110", "101", "110"),
        "E": ("111", "100", "110", "100", "111"),
        "H": ("101", "101", "111", "101", "101"),
        "P": ("110", "101", "110", "100", "100"),
        "S": ("111", "100", "111", "001", "111"),
        "V": ("101", "101", "101", "101", "010"),
        "W": ("101", "101", "101", "111", "101"),
        "X": ("101", "101", "010", "101", "101"),
        "Y": ("101", "101", "010", "010", "010"),
        "?": ("110", "001", "010", "000", "010"),
    }
    glyph_width = 3
    glyph_gap = 1
    bucket_top = banner_height - 9
    bucket_bottom = banner_height - 2

    for index, (bucket, count, baseline, label) in enumerate(
        zip(bucket_indices, bucket_counts, baseline_indices, labels)
    ):
        left, right = edges[index], edges[index + 1]
        width = right - left
        count = max(int(count), 1)
        bucket = int(np.clip(bucket, 0, count - 1))
        baseline = int(np.clip(baseline, 0, count - 1))
        changed = bucket != baseline

        if bucket < baseline:
            max_distance = max(baseline, 1)
            distance_fraction = (baseline - bucket) / max_distance
            text_color = np.rint(
                lower_near_color
                + distance_fraction * (lower_far_color - lower_near_color)
            ).astype(np.uint8)
        elif bucket > baseline:
            max_distance = max(count - 1 - baseline, 1)
            distance_fraction = (bucket - baseline) / max_distance
            text_color = np.rint(
                higher_near_color
                + distance_fraction * (higher_far_color - higher_near_color)
            ).astype(np.uint8)
        else:
            text_color = baseline_tick_color

        frames[:, :banner_height, left:right, :] = panel_color
        label = label[:2].upper().ljust(2, "?")
        label_width = len(label) * glyph_width + (len(label) - 1) * glyph_gap
        x = left + max((width - label_width) // 2, 0)
        for character in label:
            glyph = glyphs.get(character, glyphs["?"])
            for row, row_bits in enumerate(glyph):
                for column, bit in enumerate(row_bits):
                    if bit == "1":
                        frames[:, 2 + row, x + column, :] = text_color
            x += glyph_width + glyph_gap

        frames[:, bucket_top:bucket_bottom, left:right, :] = rail_color

        # The base-game marker is deliberately fixed in the visual centre of
        # every parameter section.  A changed bucket is placed as a signed,
        # normalised distance from that centre rather than at its absolute
        # bucket position, whose base index differs between parameters.
        baseline_x = left + max((width - 1) // 2, 0)
        if bucket < baseline:
            left_span = max(baseline_x - left, 1)
            pixel_offset = max(1, int(np.ceil(distance_fraction * left_span)))
            selected_x = max(left, baseline_x - pixel_offset)
        elif bucket > baseline:
            right_span = max(right - 1 - baseline_x, 1)
            pixel_offset = max(1, int(np.ceil(distance_fraction * right_span)))
            selected_x = min(right - 1, baseline_x + pixel_offset)
        else:
            selected_x = baseline_x
        if changed:
            connector_left = min(selected_x, baseline_x)
            connector_right = max(selected_x, baseline_x) + 1
            frames[
                :, bucket_bottom - 2 : bucket_bottom, connector_left:connector_right, :
            ] = text_color
            frames[:, bucket_top:bucket_bottom, selected_x : selected_x + 1, :] = text_color
        frames[:, bucket_top - 1 : bucket_bottom + 1, baseline_x : baseline_x + 1, :] = baseline_tick_color

    for edge in edges[1:-1]:
        frames[:, :banner_height, max(edge - 1, 0) : edge, :] = separator_color


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _float_tuple_config(config, key: str, default: Sequence[float]) -> tuple[float, ...]:
    """Read a YAML list as a static float tuple suitable for JAX tracing."""
    return tuple(float(value) for value in config.get(key, default))


def _make_pong_accel_wrapper(atari_env, config, level_encoding: str):
    return PongAccelModWrapper(
        atari_env,
        player_paddle_height_multiplier=float(
            config.get("ACCEL_PLAYER_PADDLE_HEIGHT_MULTIPLIER", 1.0)
        ),
        player_paddle_shrink_multiplier=float(
            config.get("ACCEL_PLAYER_PADDLE_SHRINK_MULTIPLIER", 0.5)
        ),
        player_paddle_width_multiplier=float(
            config.get("ACCEL_PLAYER_PADDLE_WIDTH_MULTIPLIER", 0.5)
        ),
        enemy_step_multiplier=float(config.get("ACCEL_ENEMY_STEP_MULTIPLIER", 2.0)),
        enemy_move_interval=int(config.get("ACCEL_ENEMY_MOVE_INTERVAL", 4)),
        enable_lazy_enemy=bool(config.get("ACCEL_ENABLE_LAZY_ENEMY", True)),
        enable_enemy_move_interval=bool(
            config.get("ACCEL_ENABLE_ENEMY_MOVE_INTERVAL", True)
        ),
        enemy_match_ball_speed=bool(
            config.get("ACCEL_ENEMY_MATCH_BALL_SPEED", False)
        ),
        randomize_serve_direction=bool(
            config.get("ACCEL_RANDOMIZE_SERVE_DIRECTION", False)
        ),
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
            (8, 10, 12, 14, 16),
        ),
        parameter_paddle_widths=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_PADDLE_WIDTHS",
            (2, 3, 4),
        ),
        parameter_enemy_step_sizes=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_ENEMY_STEP_SIZES",
            (1, 2, 3, 4),
        ),
        parameter_enemy_tracking_modes=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_ENEMY_TRACKING_MODES",
            (0, 1),
        ),
        parameter_enemy_move_intervals=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_ENEMY_MOVE_INTERVALS",
            (0, 2, 4),
        ),
    )


def _make_freeway_accel_wrapper(atari_env, config, level_encoding: str):
    return FreewayAccelModWrapper(
        atari_env,
        level_encoding=level_encoding,
        parameter_car_speed_multipliers=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_CAR_SPEED_MULTIPLIERS",
            (1.0, 1.5, 2.0),
        ),
        parameter_car_widths=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_CAR_WIDTHS",
            (8, 10, 12),
        ),
        parameter_car_heights=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_CAR_HEIGHTS",
            (10, 12, 14),
        ),
    )


def _make_seaquest_accel_wrapper(atari_env, config, level_encoding: str):
    return SeaquestAccelModWrapper(
        atari_env,
        level_encoding=level_encoding,
        parameter_player_move_speeds=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_PLAYER_MOVE_SPEEDS",
            (1, 2),
        ),
        parameter_enemy_speed_multipliers=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_ENEMY_SPEED_MULTIPLIERS",
            (1, 2),
        ),
        parameter_player_missile_speeds=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_PLAYER_MISSILE_SPEEDS",
            (3, 4, 5, 6),
        ),
        parameter_enemy_missile_speed_multipliers=_float_tuple_config(
            config,
            "ACCEL_PARAMETER_ENEMY_MISSILE_SPEED_MULTIPLIERS",
            (1, 2),
        ),
    )


def _make_kangaroo_accel_wrapper(atari_env, config, level_encoding: str):
    del level_encoding
    initial_edits = int(config.get("ACCEL_KANGAROO_INITIAL_EDITS", 3))
    mutation_edits = int(
        config.get(
            "ACCEL_KANGAROO_MUTATION_EDITS",
            config.get("ACCEL_NUM_EDITS", 1),
        )
    )
    object_step = int(config.get("ACCEL_KANGAROO_OBJECT_STEP", 8))
    config["ACCEL_KANGAROO_INITIAL_EDITS"] = initial_edits
    config["ACCEL_KANGAROO_MUTATION_EDITS"] = mutation_edits
    config["ACCEL_KANGAROO_OBJECT_STEP"] = object_step
    return KangarooAccelWrapper(
        atari_env,
        initial_edits=initial_edits,
        mutation_edits=mutation_edits,
        object_step=object_step,
    )


ACCEL_WRAPPER_FACTORIES = {
    "freeway": _make_freeway_accel_wrapper,
    "kangaroo": _make_kangaroo_accel_wrapper,
    "pong": _make_pong_accel_wrapper,
    "seaquest": _make_seaquest_accel_wrapper,
}


def _unwrap_state_for_render(state):
    """Reach the raw JAXAtari game state from the normal wrapper stack."""
    while True:
        if hasattr(state, "atari_state"):
            state = state.atari_state
        elif hasattr(state, "env_state"):
            state = state.env_state
        else:
            return state


def _prepare_video_frames(frames: np.ndarray) -> np.ndarray:
    """Convert rendered video frames to W&B-friendly RGB without changing observations."""
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames shaped (T, H, W, C), got {frames.shape}.")
    if frames.shape[-1] == 1:
        return np.repeat(frames, 3, axis=-1)
    if frames.shape[-1] >= 3:
        return frames[..., :3]
    raise ValueError(f"Video frames must have one or at least three channels, got {frames.shape}.")


def train_state_to_log_dict(train_state: CustomTrainState, level_sampler: LevelSampler) -> dict:
    """JAXUED-style lightweight sampler logging for mask levels."""
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)
    # LevelSampler deliberately stores -inf in unused score slots. Mask those
    # values before arithmetic: multiplying an unused -inf score by its zero
    # sampling weight produces NaN and corrupts only the logging metric.
    valid_scores = jnp.where(idx, sampler["scores"], 0.0)
    weights = level_sampler.level_weights(sampler)
    return {
        "level_sampler/size": sampler["size"],
        "level_sampler/episode_count": sampler["episode_count"],
        "level_sampler/max_score": jnp.max(
            jnp.where(idx, sampler["scores"], -jnp.inf)
        ),
        "level_sampler/weighted_score": (valid_scores * weights).sum(),
        "level_sampler/mean_score": valid_scores.sum() / s,
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


def _reachable_mask_level_count(
    num_actions: int,
    max_level_size: int,
    conflicting_pairs: tuple[tuple[int, int], ...],
) -> int:
    """Count valid mask levels exactly, including mutually exclusive actions."""
    max_level_size = min(max(max_level_size, 0), num_actions)
    unique_conflicts = {
        tuple(sorted((left, right)))
        for left, right in conflicting_pairs
        if left != right
    }
    return sum(
        level.bit_count() <= max_level_size
        and not any(
            level & (1 << left) and level & (1 << right)
            for left, right in unique_conflicts
        )
        for level in range(1 << num_actions)
    )


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
    """Sample a quantized level, occasionally preserving the exact base game."""
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
    """Move one, or occasionally two, mutable parameters by exactly one bin."""
    level = jnp.asarray(level, dtype=jnp.int32)
    bin_counts = jnp.asarray(bin_counts, dtype=jnp.int32)
    n = level.shape[0]
    rng_indices, rng_signs, rng_multi = jax.random.split(rng, 3)
    mutable_dimensions = bin_counts > 1
    mutable_count = jnp.sum(mutable_dimensions)
    # Randomly order only dimensions that have somewhere to move. Immutable
    # one-bin dimensions sort to the end and are never selected.
    priorities = jnp.where(
        mutable_dimensions,
        jax.random.uniform(rng_indices, shape=(n,)),
        -jnp.inf,
    )
    indices = jnp.argsort(priorities)[::-1]
    signs = jax.random.choice(
        rng_signs, jnp.asarray([-1, 1], dtype=jnp.int32), shape=(2,)
    )
    mutate_second = (
        jax.random.bernoulli(rng_multi, multi_param_probability)
        & (mutable_count > 1)
    )

    def mutate_one(values, index, sign):
        last_bin = bin_counts[index] - 1
        # Force boundary mutations inward instead of clipping them back to the
        # original bucket, which previously made many mutations exact no-ops.
        direction = jnp.where(
            values[index] <= 0,
            1,
            jnp.where(values[index] >= last_bin, -1, sign),
        )
        return values.at[index].set(
            values[index] + direction
        )

    mutated = jax.lax.cond(
        mutable_count > 0,
        lambda values: mutate_one(values, indices[0], signs[0]),
        lambda values: values,
        level,
    )
    return jax.lax.cond(
        mutate_second,
        lambda values: mutate_one(values, indices[1], signs[1]),
        lambda values: values,
        mutated,
    )


def _compute_mask_scores(
    config,
    dones,
    rewards,
    values,
    lambda_targets,
    env_rewards=None,
):
    """Compute stateless level scores, with larger values prioritized more.

    Dense Q-PVL is the default PQN regret proxy. ``env_return`` remains a raw
    difficulty baseline, while stateful ``env_learning_progress`` is applied
    by the sampler update path because it needs the level's previous EMA.
    """
    score_fn = str(
        config.get("JAXUED_SCORE_FUNCTION", "q_pvl")
    ).lower()
    incomplete_value = float(config.get("JAXUED_INCOMPLETE_SCORE", 0.0))

    if score_fn in {
        "env_return",
        "negative_env_return",
        "reward_difficulty",
        "env_reward",
    }:
        score_rewards = rewards if env_rewards is None else env_rewards
        return compute_env_return_score(
            score_rewards,
            higher_is_better=bool(
                config.get("JAXUED_ENV_REWARD_HIGHER_IS_BETTER", True)
            ),
        )
    if score_fn in {"pvl", "jaxued_pvl"}:
        return positive_value_loss(
            dones,
            lambda_targets - values,
            incomplete_value=incomplete_value,
        )
    if score_fn in {"max_mc", "maxmc"}:
        max_returns = compute_max_returns(dones, rewards)
        return max_mc(
            dones,
            values,
            max_returns,
            incomplete_value=incomplete_value,
        )
    if score_fn in {
        "q_pvl",
        "dense_q_pvl",
        "q_pvl_all_steps",
        "dense_pvl",
    }:
        return compute_dense_q_pvl(values, lambda_targets)
    raise ValueError(
        "JAXUED_SCORE_FUNCTION must be one of "
        "'q_pvl', 'env_learning_progress', 'env_return', 'pvl', "
        "or 'max_mc'."
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
    accel_mutation_prob = float(config.get("ACCEL_MUTATION_PROB", 1.0))
    if not 0.0 <= accel_mutation_prob <= 1.0:
        raise ValueError("ACCEL_MUTATION_PROB must be in [0, 1].")
    protagonist_alg = str(config.get("PROTAGONIST_ALG", "pqn")).lower()
    if protagonist_alg not in {"pqn", "jaxued_ppo"}:
        raise ValueError(f"PROTAGONIST_ALG must be 'pqn' or 'jaxued_ppo', got {protagonist_alg!r}")

    train_mods = config.get("TRAIN_MODS", None)
    train_mods_list = train_mods if isinstance(train_mods, list) else [train_mods] if train_mods else None

    env_name = str(config["ENV_NAME"]).lower()
    default_adv_wrapper = (
        env_name[: -len("_default")]
        if env_name.endswith("_default")
        else env_name
    )
    configured_adv_wrapper = str(
        config.get("ADV_WRAPPER", default_adv_wrapper)
    ).lower()
    # A registered game selects its own adapter. This also prevents a copied
    # Pong/Seaquest config with a stale ADV_WRAPPER from wrapping Kangaroo with
    # the wrong state schema. Explicit adapters remain available to unregistered
    # compatible environment aliases.
    adv_wrapper = (
        default_adv_wrapper
        if default_adv_wrapper in ACCEL_WRAPPER_FACTORIES
        else configured_adv_wrapper
    )
    config["ADV_WRAPPER"] = adv_wrapper
    wrapper_factory = ACCEL_WRAPPER_FACTORIES.get(adv_wrapper)
    if wrapper_factory is None:
        supported = ", ".join(sorted(ACCEL_WRAPPER_FACTORIES))
        raise ValueError(
            f"Unsupported ADV_WRAPPER={adv_wrapper!r}; choose one of: {supported}."
        )
    requested_level_encoding = str(
        config.get("ACCEL_LEVEL_ENCODING", "mask")
    ).lower()
    if requested_level_encoding not in {"mask", "parameter_vector", "structured"}:
        raise ValueError(
            "ACCEL_LEVEL_ENCODING must be 'mask', 'parameter_vector', or "
            "'structured', got "
            f"{requested_level_encoding!r}"
        )

    def apply_accel_wrapper(atari_env):
        """Place ACCEL directly above AtariWrapper so it runs every raw frame."""
        return wrapper_factory(atari_env, config, requested_level_encoding)

    def apply_wrappers(base_env, *, use_accel: bool = False):
        wrapped = AtariWrapper(base_env)
        if use_accel:
            wrapped = apply_accel_wrapper(wrapped)
        # PixelObsWrapper's native-downscaling mode replaces the core renderer
        # with the learner's small grayscale renderer. Retain the original
        # full-resolution RGB renderer solely for rollout videos.
        video_renderer = wrapped.renderer
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
        wrapped._video_renderer = video_renderer
        wrapped = NormalizeObservationWrapper(wrapped)
        wrapped = LogWrapper(wrapped)
        return wrapped

    env = apply_wrappers(
        jaxatari.make(config["ENV_NAME"].lower(), mods=train_mods_list),
        use_accel=True,
    )
    # The wrapper is authoritative. Structured games can therefore override a
    # legacy encoding left in a shared config without a game-specific runner
    # branch; mask/parameter wrappers retain the requested value.
    level_encoding = str(
        getattr(env, "_level_encoding", requested_level_encoding)
    ).lower()
    if level_encoding not in {"mask", "parameter_vector", "structured"}:
        raise ValueError(
            f"ACCEL wrapper returned unsupported level encoding {level_encoding!r}."
        )
    config["ACCEL_LEVEL_ENCODING"] = level_encoding
    if wandb.run is not None:
        resolved_config = {
            "ADV_WRAPPER": adv_wrapper,
            "ACCEL_LEVEL_ENCODING": level_encoding,
        }
        resolved_config.update(
            {
                key: config[key]
                for key in (
                    "ACCEL_KANGAROO_INITIAL_EDITS",
                    "ACCEL_KANGAROO_MUTATION_EDITS",
                    "ACCEL_KANGAROO_OBJECT_STEP",
                )
                if key in config
            }
        )
        wandb.config.update(
            resolved_config,
            allow_val_change=True,
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
    eval_seed = config.get("EVAL_SEED", None)
    if eval_seed is not None:
        eval_seed = int(eval_seed)
    default_eval_mods = {
        "freeway": ["slow_cars", "stop_all_cars"],
        "pong": ["lazy_enemy"],
    }.get(adv_wrapper, [])
    eval_mods = _as_list(
        config.get("EVAL_MODS", config.get("MOD_NAME", default_eval_mods))
    )
    eval_record_video = bool(config.get("EVAL_RECORD_VIDEO", config.get("RECORD_VIDEO", False)))
    eval_video_base = bool(config.get("EVAL_VIDEO_BASE", True))
    eval_video_mods = bool(config.get("EVAL_VIDEO_MODS", True))
    eval_video_max_steps = int(config.get("EVAL_VIDEO_MAX_STEPS", 1_000))
    eval_video_fps = int(config.get("EVAL_VIDEO_FPS", 30))
    accel_video_overlay_height = int(config.get("ACCEL_VIDEO_OVERLAY_HEIGHT", 32))
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

    structured_levels = bool(getattr(env, "uses_structured_levels", False))
    if structured_levels != (level_encoding == "structured"):
        raise ValueError(
            "ACCEL wrapper and configured level encoding disagree about "
            "structured-level support."
        )

    def spec_display_name(spec):
        return getattr(spec, "label", None) or spec.target

    if structured_levels:
        mod_action_space_n = 0
        mod_spec_metric_names = []
        parameter_value_names = []
        accel_short_labels = []
        conflicting_mask_pairs = ()
        max_level_size = 0
    else:
        if not hasattr(env, "_mod_specs"):
            raise ValueError("ACCEL wrapper missing _mod_specs")
        mod_action_space_n = len(env._mod_specs)
        mod_spec_metric_names = [
            f"accel_actions/spec_{i:02d}_{_safe_metric_fragment(spec_display_name(spec))}"
            for i, spec in enumerate(env._mod_specs)
        ]
        parameter_value_names = list(env.accel_parameter_names)
        accel_short_labels = list(env.accel_short_labels)
        if (
            len(parameter_value_names) != mod_action_space_n
            or len(accel_short_labels) != mod_action_space_n
        ):
            raise ValueError(
                f"{env.accel_game_name} ACCEL metadata no longer matches wrapper dimensions."
            )
        conflicting_mask_pairs = tuple(getattr(env, "_conflicting_mask_pairs", ()))
        max_level_size = min(
            int(config.get("ACCEL_MAX_LEVEL_SIZE", mod_action_space_n)),
            mod_action_space_n,
        )
    parameter_metric_names = [
        f"accel_parameters/{_safe_metric_fragment(name)}"
        for name in parameter_value_names[:mod_action_space_n]
    ]
    duplicate_check = bool(
        config.get("BUFFER_DUPLICATE_CHECK", config.get("buffer_duplicate_check", True))
    )
    score_function = str(config.get("JAXUED_SCORE_FUNCTION", "q_pvl")).lower()
    reward_progress_score = score_function in {
        "env_learning_progress",
        "reward_learning_progress",
        "reward_progress",
    }
    reward_progress_ema_alpha = float(
        config.get("JAXUED_REWARD_PROGRESS_EMA_ALPHA", 0.2)
    )
    reward_progress_new_level_score = float(
        config.get("JAXUED_REWARD_PROGRESS_NEW_LEVEL_SCORE", 1.0)
    )
    reward_progress_epsilon = float(
        config.get("JAXUED_REWARD_PROGRESS_EPSILON", 1e-6)
    )
    if reward_progress_score and not duplicate_check:
        raise ValueError(
            "env_learning_progress requires BUFFER_DUPLICATE_CHECK=true so "
            "each exact level has one persistent return history."
        )
    if not 0.0 < reward_progress_ema_alpha <= 1.0:
        raise ValueError("JAXUED_REWARD_PROGRESS_EMA_ALPHA must be in (0, 1].")
    if reward_progress_new_level_score < 0.0:
        raise ValueError(
            "JAXUED_REWARD_PROGRESS_NEW_LEVEL_SCORE must be non-negative."
        )
    if reward_progress_epsilon <= 0.0:
        raise ValueError("JAXUED_REWARD_PROGRESS_EPSILON must be positive.")
    requested_sampler_capacity = int(
        config.get("ACCEL_BUFFER_CAPACITY", config.get("LEVEL_BUFFER_CAPACITY", 64))
    )
    if structured_levels:
        reachable_level_count = None
    elif level_encoding == "parameter_vector":
        parameter_bin_counts = env.parameter_bin_counts
        parameter_baseline_level = env.parameter_baseline_level
        reachable_level_count = int(np.prod(np.asarray(parameter_bin_counts)))
    else:
        reachable_level_count = _reachable_mask_level_count(
            mod_action_space_n,
            max_level_size,
            conflicting_mask_pairs,
        )
    sampler_capacity = (
        requested_sampler_capacity
        if structured_levels or not duplicate_check
        else min(requested_sampler_capacity, reachable_level_count)
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

    if structured_levels:
        placeholder_level = env.baseline_level
        placeholder_batch = jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                leaf, (config["NUM_ENVS"], *jnp.shape(leaf))
            ),
            placeholder_level,
        )
    elif level_encoding == "parameter_vector":
        placeholder_level = parameter_baseline_level
        placeholder_batch = jnp.broadcast_to(
            placeholder_level, (config["NUM_ENVS"], mod_action_space_n)
        )
    else:
        placeholder_level = jnp.zeros((mod_action_space_n,), dtype=bool)
        placeholder_batch = jnp.zeros(
            (config["NUM_ENVS"], mod_action_space_n), dtype=bool
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
        level_extra = (
            {
                "return_ema": jnp.asarray(0.0, dtype=jnp.float32),
                "return_initialized": jnp.asarray(False, dtype=bool),
            }
            if reward_progress_score
            else None
        )
        sampler = level_sampler.initialize(placeholder_level, level_extra)
        return CustomTrainState.create(
            apply_fn=network.apply,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            sampler=sampler,
            update_state=jnp.array(UpdateState.DR, dtype=jnp.int32),
            last_replay_level_batch=placeholder_batch,
            last_replay_level_inds=jnp.full(
                (config["NUM_ENVS"],), -1, dtype=jnp.int32
            ),
            dr_last_level_batch=placeholder_batch,
            mutation_last_level_batch=placeholder_batch,
            current_level_batch=placeholder_batch,
        )

    def _progress_score(previous_return_ema, return_initialized, env_return):
        return compute_reward_learning_progress_score(
            env_return,
            previous_return_ema,
            return_initialized,
            ema_alpha=reward_progress_ema_alpha,
            new_level_score=reward_progress_new_level_score,
            epsilon=reward_progress_epsilon,
        )

    def insert_progress_batch(sampler, levels, env_returns):
        """Insert novel levels and update duplicate levels sequentially.

        Sequential processing matters for batches containing the same level
        more than once (notably the baseline): every observed return contributes
        to that level's EMA instead of the last environment silently winning.
        """

        def _insert_one(current_sampler, level_and_return):
            level, env_return = level_and_return
            level_idx = level_sampler.find(current_sampler, level)
            has_history = level_idx >= 0
            safe_idx = jnp.maximum(level_idx, 0)
            previous_return_ema = jnp.where(
                has_history,
                current_sampler["levels_extra"]["return_ema"][safe_idx],
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            return_initialized = jnp.where(
                has_history,
                current_sampler["levels_extra"]["return_initialized"][safe_idx],
                jnp.asarray(False, dtype=bool),
            )
            score, updated_return_ema = _progress_score(
                previous_return_ema,
                return_initialized,
                env_return,
            )
            updated_sampler, inserted_idx = level_sampler.insert(
                current_sampler,
                level,
                score,
                {
                    "return_ema": updated_return_ema,
                    "return_initialized": jnp.asarray(True, dtype=bool),
                },
            )
            return updated_sampler, (score, inserted_idx)

        return jax.lax.scan(_insert_one, sampler, (levels, env_returns))

    def update_progress_batch(sampler, level_inds, env_returns):
        """Update reward progress for already sampled replay entries."""

        def _update_one(current_sampler, index_and_return):
            level_idx, env_return = index_and_return
            previous_return_ema = current_sampler["levels_extra"]["return_ema"][
                level_idx
            ]
            return_initialized = current_sampler["levels_extra"][
                "return_initialized"
            ][level_idx]
            score, updated_return_ema = _progress_score(
                previous_return_ema,
                return_initialized,
                env_return,
            )
            updated_sampler = level_sampler.update(
                current_sampler,
                level_idx,
                score,
                {
                    "return_ema": updated_return_ema,
                    "return_initialized": jnp.asarray(True, dtype=bool),
                },
            )
            return updated_sampler, score

        return jax.lax.scan(_update_one, sampler, (level_inds, env_returns))

    def level_round_metrics(levels, strength):
        """Log decoded parameter values or legacy mask occupancy per update."""
        if structured_levels:
            return env.level_metrics(levels)
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

        lambda_targets = compute_q_lambda_targets(
            transitions.reward,
            transitions.done,
            transitions.q_val,
            last_q,
            config["GAMMA"],
            config["LAMBDA"],
        )

        chosen_q = jnp.take_along_axis(
            transitions.q_val,
            transitions.action[..., None],
            axis=-1,
        ).squeeze(-1)
        # PLR scores policy-level prediction error, not Q magnitude. For an
        # epsilon-greedy PQN, max_a Q(s, a) is the critic baseline for the
        # protagonist's greedy target policy.
        state_values = jnp.max(transitions.q_val, axis=-1)
        greedy_actions = jnp.argmax(transitions.q_val, axis=-1)
        raw_env_rewards = infos.get("env_reward", transitions.reward)
        env_returns = compute_env_returns(raw_env_rewards)
        scores = (
            jnp.zeros_like(env_returns)
            if reward_progress_score
            else _compute_mask_scores(
                config,
                transitions.done,
                transitions.reward,
                state_values,
                lambda_targets,
                env_rewards=raw_env_rewards,
            )
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
            "level_sampler/env_return_mean": jnp.mean(env_returns),
            "qvals": qvals_mean,
            "level_sampler/score_mean": jnp.mean(scores),
            **_mean_infos(
                infos,
                skip=(
                    "level_mask",
                    "level_parameters",
                    "parameter_values",
                    "strength",
                    "returned_episode_returns",
                    "returned_episode_lengths",
                ),
            ),
            **_completed_episode_metrics(infos),
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
        return (
            train_state,
            rng,
            last_obs,
            final_env_state,
            scores,
            env_returns,
            round_metrics,
        )

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

        raw_env_rewards = infos.get("env_reward", rewards)
        env_returns = compute_env_returns(raw_env_rewards)
        scores = (
            jnp.zeros_like(env_returns)
            if reward_progress_score
            else _compute_mask_scores(
                config,
                dones,
                rewards,
                values,
                targets,
                env_rewards=raw_env_rewards,
            )
        )

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
            "level_sampler/env_return_mean": jnp.mean(env_returns),
            "rollout_reward_mean": jnp.mean(rewards),
            **_mean_infos(
                infos,
                skip=(
                    "level_mask",
                    "level_parameters",
                    "parameter_values",
                    "strength",
                    "returned_episode_returns",
                    "returned_episode_lengths",
                ),
            ),
            **_completed_episode_metrics(infos),
        }
        round_metrics.update(level_round_metrics(level_masks, strength))
        return (
            train_state,
            rng,
            last_obs,
            final_env_state,
            scores,
            env_returns,
            round_metrics,
        )

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
        mask, parameter vector, or structured level used by the curriculum.
        """
        max_steps = eval_video_max_steps if max_steps is None else max_steps
        if max_steps <= 0:
            return

        if level_mask is not None and getattr(
            rollout_env, "uses_structured_levels", False
        ):
            obs, env_state = rollout_env.reset(rng, level_mask)
        else:
            obs, env_state = rollout_env.reset(rng)
        renderer = getattr(rollout_env, "_video_renderer", rollout_env.renderer)
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
        frames = _prepare_video_frames(np.stack(frames, axis=0))

        # The top bar identifies dimensions that differ from the base game.
        # In parameter-vector mode the W&B caption and scalar metrics contain
        # exact decoded values; the highlighted panels show which dimensions
        # are currently non-baseline directly in the video pixels.
        metrics_payload = {
            "num_env_steps": env_step,
            f"video_metrics/{label}/return": total_reward,
            f"video_metrics/{label}/length": len(frames),
        }
        if level_mask is not None:
            is_structured_level = bool(
                getattr(rollout_env, "uses_structured_levels", False)
            )
            if is_structured_level:
                batched_level = jax.tree_util.tree_map(
                    lambda leaf: leaf[None, ...], level_mask
                )
                structured_metrics = rollout_env.level_metrics(batched_level)
                for metric_name, value in structured_metrics.items():
                    metrics_payload[
                        f"video_metrics/{label}/{_safe_metric_fragment(metric_name)}"
                    ] = float(jax.device_get(value))
                video_caption = rollout_env.describe_level(level_mask)
            else:
                is_parameter_level = (
                    getattr(rollout_env, "_level_encoding", "mask")
                    == "parameter_vector"
                )
                raw_level = np.asarray(jax.device_get(level_mask))
                if is_parameter_level:
                    mask = raw_level != np.asarray(
                        jax.device_get(rollout_env.parameter_baseline_level)
                    )
                else:
                    mask = raw_level.astype(bool)
                action_targets = [
                    spec_display_name(spec) for spec in rollout_env._mod_specs
                ]
                action_labels = accel_short_labels
                if is_parameter_level:
                    _overlay_accel_bucket_banner(
                        frames,
                        raw_level,
                        np.asarray(jax.device_get(rollout_env.parameter_bin_counts)),
                        np.asarray(
                            jax.device_get(rollout_env.parameter_baseline_level)
                        ),
                        action_labels,
                        height=accel_video_overlay_height,
                    )
                    decoded = np.asarray(
                        jax.device_get(
                            rollout_env.decode_parameter_level(level_mask)
                        )
                    )
                    display_names = parameter_value_names[: len(decoded)]
                    for index, value in enumerate(decoded):
                        metric_name = _safe_metric_fragment(display_names[index])
                        metrics_payload[
                            f"video_metrics/{label}/parameter/{metric_name}"
                        ] = float(value)
                    metrics_payload[
                        f"video_metrics/{label}/changed_dimensions"
                    ] = int(mask.sum())
                    video_caption = (
                        "ACCEL parameters: "
                        + ", ".join(
                            f"{name}={value:g}"
                            for name, value in zip(display_names, decoded)
                        )
                        + " | top strip: white=center/base game; "
                        "blue=lower, warm=higher"
                    )
                else:
                    _overlay_accel_mask_banner(
                        frames,
                        mask,
                        action_labels,
                        height=accel_video_overlay_height,
                    )
                    for index, active in enumerate(mask):
                        metric_name = mod_spec_metric_names[index].split("/")[-1]
                        metrics_payload[
                            f"video_metrics/{label}/mask/{metric_name}"
                        ] = float(active)
                    metrics_payload[f"video_metrics/{label}/active_bits"] = int(
                        mask.sum()
                    )
                    metrics_payload[f"video_metrics/{label}/strength"] = float(
                        strength
                    )
                    active_targets = [
                        target
                        for target, active in zip(action_targets, mask)
                        if active
                    ]
                    video_caption = (
                        "ACCEL active: "
                        f"{', '.join(active_targets) if active_targets else 'none'} "
                        f"| mask: {''.join('1' if active else '0' for active in mask)} "
                        "| legend: "
                        + ", ".join(
                            f"{short}={target}"
                            for short, target in zip(action_labels, action_targets)
                        )
                    )
        else:
            video_caption = "Evaluation rollout without an ACCEL level."
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
            if eval_seed is not None:
                # Use the same reset/no-op/sticky-action streams as the plain
                # baseline so evaluation differences reflect the policy.
                rng = jax.random.PRNGKey(eval_seed)

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
            episode_count = jnp.sum(returned_f)
            denom = jnp.maximum(episode_count, 1.0)

            def _masked_mean(x):
                x = x.squeeze()
                if x.dtype == jnp.bool_:
                    x = x.astype(jnp.float32)
                mask = jnp.broadcast_to(returned, x.shape)
                mean = jnp.sum(jnp.where(mask, x, 0.0)) / denom
                return jnp.where(
                    episode_count > 0,
                    mean,
                    jnp.asarray(jnp.nan, dtype=jnp.float32),
                )

            done_infos = jax.tree_util.tree_map(_masked_mean, infos)
            return {
                "returned_episode_returns": done_infos["returned_episode_returns"],
                "returned_episode_lengths": done_infos["returned_episode_lengths"],
                "episodes": episode_count,
            }

        return _eval

    eval_fns = [
        (label, _build_eval_fn(eval_env), eval_env)
        for label, eval_env in eval_env_entries
    ]

    def reset_rollout_for_levels(rng, levels, rollout_obs, rollout_env_state):
        """Give structured layouts Maze-style reset-to-level semantics."""
        if not structured_levels:
            return rng, rollout_obs, rollout_env_state
        rng, reset_rng = jax.random.split(rng)
        rollout_obs, rollout_env_state = jax.vmap(env.reset)(
            jax.random.split(reset_rng, config["NUM_ENVS"]),
            levels,
        )
        return rng, rollout_obs, rollout_env_state

    @jax.jit
    def train_step(carry, _):
        rng, train_state, rollout_obs, rollout_env_state = carry
        strength = jnp.asarray(config.get("ACCEL_STRENGTH", 1.0), dtype=jnp.float32)

        def random_level_batch(rng):
            if structured_levels:
                return jax.vmap(env.sample_random_level)(
                    jax.random.split(rng, config["NUM_ENVS"])
                )
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

        def on_new_levels(rng, train_state, rollout_obs, rollout_env_state):
            rng, rng_levels, rng_rollout = jax.random.split(rng, 3)
            levels = random_level_batch(rng_levels)
            rng_rollout, rollout_obs, rollout_env_state = reset_rollout_for_levels(
                rng_rollout, levels, rollout_obs, rollout_env_state
            )
            (
                train_state,
                rng_rollout,
                next_obs,
                next_env_state,
                scores,
                env_returns,
                round_metrics,
            ) = train_round(
                train_state,
                rollout_obs,
                rollout_env_state,
                rng_rollout,
                levels,
                strength,
                bool(config.get("EXPLORATORY_GRAD_UPDATES", True)),
            )
            if reward_progress_score:
                sampler, (scores, _) = insert_progress_batch(
                    train_state.sampler,
                    levels,
                    env_returns,
                )
            else:
                sampler, _ = level_sampler.insert_batch(
                    train_state.sampler,
                    levels,
                    scores,
                )
            round_metrics = {
                **round_metrics,
                "level_sampler/score_mean": jnp.mean(scores),
            }
            train_state = train_state.replace(
                sampler=sampler,
                update_state=jnp.array(UpdateState.DR, dtype=jnp.int32),
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=levels,
                current_level_batch=levels,
            )
            return (rng_rollout, train_state, next_obs, next_env_state), round_metrics

        def on_replay_levels(rng, train_state, rollout_obs, rollout_env_state):
            rng, rng_levels, rng_rollout = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                train_state.sampler,
                rng_levels,
                config["NUM_ENVS"],
            )
            train_state = train_state.replace(sampler=sampler)
            rng_rollout, rollout_obs, rollout_env_state = reset_rollout_for_levels(
                rng_rollout, levels, rollout_obs, rollout_env_state
            )
            (
                train_state,
                rng_rollout,
                next_obs,
                next_env_state,
                scores,
                env_returns,
                round_metrics,
            ) = train_round(
                train_state,
                rollout_obs,
                rollout_env_state,
                rng_rollout,
                levels,
                strength,
                True,
            )
            if reward_progress_score:
                sampler, scores = update_progress_batch(
                    train_state.sampler,
                    level_inds,
                    env_returns,
                )
            else:
                sampler = level_sampler.update_batch(
                    train_state.sampler,
                    level_inds,
                    scores,
                )
            round_metrics = {
                **round_metrics,
                "level_sampler/score_mean": jnp.mean(scores),
            }
            train_state = train_state.replace(
                sampler=sampler,
                update_state=jnp.array(UpdateState.REPLAY, dtype=jnp.int32),
                num_replay_updates=train_state.num_replay_updates + 1,
                last_replay_level_batch=levels,
                last_replay_level_inds=level_inds,
                current_level_batch=levels,
            )
            return (rng_rollout, train_state, next_obs, next_env_state), round_metrics

        def on_mutate_levels(rng, train_state, rollout_obs, rollout_env_state):
            rng, rng_mutate, rng_rollout = jax.random.split(rng, 3)
            parent_levels = jax.lax.cond(
                train_state.num_replay_updates > 0,
                lambda: train_state.last_replay_level_batch,
                lambda: placeholder_batch,
            )
            if structured_levels:
                levels = jax.vmap(env.mutate_level)(
                    jax.random.split(rng_mutate, config["NUM_ENVS"]),
                    parent_levels,
                )
            elif level_encoding == "parameter_vector":
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
            rng_rollout, rollout_obs, rollout_env_state = reset_rollout_for_levels(
                rng_rollout, levels, rollout_obs, rollout_env_state
            )
            (
                train_state,
                rng_rollout,
                next_obs,
                next_env_state,
                scores,
                env_returns,
                round_metrics,
            ) = train_round(
                train_state,
                rollout_obs,
                rollout_env_state,
                rng_rollout,
                levels,
                strength,
                bool(config.get("EXPLORATORY_GRAD_UPDATES", True)),
            )
            if reward_progress_score:
                sampler, (scores, _) = insert_progress_batch(
                    train_state.sampler,
                    levels,
                    env_returns,
                )
            else:
                sampler, _ = level_sampler.insert_batch(
                    train_state.sampler,
                    levels,
                    scores,
                )
            round_metrics = {
                **round_metrics,
                "level_sampler/score_mean": jnp.mean(scores),
            }
            train_state = train_state.replace(
                sampler=sampler,
                update_state=jnp.array(UpdateState.DR, dtype=jnp.int32),
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=levels,
                current_level_batch=levels,
            )
            return (rng_rollout, train_state, next_obs, next_env_state), round_metrics

        def on_repeat_replay_levels(rng, train_state, rollout_obs, rollout_env_state):
            """Train again on the previous replay batch before mutating it."""
            rng, rng_rollout = jax.random.split(rng)
            levels = train_state.last_replay_level_batch
            rng_rollout, rollout_obs, rollout_env_state = reset_rollout_for_levels(
                rng_rollout, levels, rollout_obs, rollout_env_state
            )
            (
                train_state,
                rng_rollout,
                next_obs,
                next_env_state,
                scores,
                env_returns,
                round_metrics,
            ) = train_round(
                train_state,
                rollout_obs,
                rollout_env_state,
                rng_rollout,
                levels,
                strength,
                True,
            )
            # Repeats re-evaluate the exact same levels but do not insert them
            # again, avoiding artificial LevelSampler timestamp updates.
            if reward_progress_score:
                sampler, scores = update_progress_batch(
                    train_state.sampler,
                    train_state.last_replay_level_inds,
                    env_returns,
                )
            else:
                sampler = level_sampler.update_batch(
                    train_state.sampler,
                    train_state.last_replay_level_inds,
                    scores,
                )
            round_metrics = {
                **round_metrics,
                "level_sampler/score_mean": jnp.mean(scores),
            }
            train_state = train_state.replace(
                sampler=sampler,
                update_state=jnp.array(UpdateState.REPLAY, dtype=jnp.int32),
                num_replay_updates=train_state.num_replay_updates + 1,
                last_replay_level_batch=levels,
                current_level_batch=levels,
            )
            return (rng_rollout, train_state, next_obs, next_env_state), round_metrics

        rng, rng_replay, rng_mutation = jax.random.split(rng, 3)
        replay_decision = (
            level_sampler.sample_replay_decision(train_state.sampler, rng_replay)
            & (train_state.sampler["size"] > 0)
        ).astype(jnp.int32)
        if alg_mode == "accel":
            # After every replay, either mutate its levels or train on that
            # same batch again. A lower mutation probability gives the
            # protagonist more learning updates per curriculum level.
            mutation_decision = jax.random.bernoulli(
                rng_mutation,
                accel_mutation_prob,
            ).astype(jnp.int32)
            branch = jax.lax.select(
                train_state.update_state == UpdateState.REPLAY,
                3 - mutation_decision,
                replay_decision,
            )
        else:
            branch = replay_decision

        (rng, train_state, rollout_obs, rollout_env_state), metrics = jax.lax.switch(
            branch,
            [on_new_levels, on_replay_levels, on_mutate_levels, on_repeat_replay_levels],
            rng,
            train_state,
            rollout_obs,
            rollout_env_state,
        )
        metrics = {
            **metrics,
            "updates/dr": train_state.num_dr_updates,
            "updates/replay": train_state.num_replay_updates,
            "updates/mutation": train_state.num_mutation_updates,
            "branch": branch,
        }
        return (rng, train_state, rollout_obs, rollout_env_state), metrics

    def train(rng):
        rng, agent_rng, rollout_reset_rng = jax.random.split(rng, 3)
        train_state = create_agent(agent_rng)
        reset_keys = jax.random.split(rollout_reset_rng, config["NUM_ENVS"])
        if structured_levels:
            rollout_obs, rollout_env_state = jax.vmap(env.reset)(
                reset_keys, placeholder_batch
            )
        else:
            rollout_obs, rollout_env_state = jax.vmap(env.reset)(reset_keys)

        metrics_history = []
        # Mask/parameter games retain the historical carried state. Structured
        # games reset inside each curriculum branch so a rollout starts on the
        # exact sampled/replayed layout, matching maze PLR/ACCEL semantics.
        runner_state = (rng, train_state, rollout_obs, rollout_env_state)
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
                # Keep W&B metric paths identical to pqn_agent.py.  Only video
                # keys need the flattened label used below.
                metric_label = label
                episode_count = float(jax.device_get(eval_metrics["episodes"]))
                snapshot_metrics[
                    f"{metric_prefix}/{metric_label}/episodes"
                ] = episode_count
                # Do not emit zero placeholders for an evaluation window in
                # which no complete episode was observed.
                if episode_count > 0:
                    snapshot_metrics.update({
                        f"{metric_prefix}/{metric_label}/returned_episode_returns": float(
                            jax.device_get(eval_metrics["returned_episode_returns"])
                        ),
                        f"{metric_prefix}/{metric_label}/returned_episode_lengths": float(
                            jax.device_get(eval_metrics["returned_episode_lengths"])
                        ),
                    })
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
            rollout_obs = runner_state[2]
            rollout_env_state = runner_state[3]
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
            round_metrics_host = _host_metrics(metrics)
            # Do not ask W&B to plot NaN or a synthetic zero while no full
            # game ended in this short rollout. ``returned_episode_count`` and
            # ``returned_episode`` still reveal that no completion occurred.
            if round_metrics_host["returned_episode_count"] <= 0:
                for metric_name in (
                    "returned_episode_returns",
                    "returned_episode_lengths",
                ):
                    round_metrics_host.pop(metric_name, None)
            metrics_host = {
                "env_step": int(jax.device_get(train_state.timesteps)),
                "update_steps": int(jax.device_get(train_state.n_updates)),
                "num_updates": int(jax.device_get(update_count)),
                "num_env_steps": int(jax.device_get(train_state.timesteps)),
                "sps": float(env_steps_per_update / elapsed),
                **round_metrics_host,
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
                        level_mask=jax.tree_util.tree_map(
                            lambda leaf: leaf[0],
                            train_state.current_level_batch,
                        ),
                        strength=float(config.get("ACCEL_STRENGTH", 1.0)),
                        max_steps=train_accel_video_max_steps,
                    )
                train_video_index += 1
            runner_state = (rng, train_state, rollout_obs, rollout_env_state)
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

        runner_state = (rng, train_state, rollout_obs, rollout_env_state)
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
        group=config.get("WANDB_GROUP"),
        job_type=config.get("WANDB_JOB_TYPE"),
        config=config,
        mode=config["WANDB_MODE"],
    )
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("eval/*", step_metric="num_env_steps")
    wandb.define_metric("video/*", step_metric="num_env_steps")
    wandb.define_metric("video_metrics/*", step_metric="num_env_steps")
    wandb.define_metric("mask/*", step_metric="num_updates")
    wandb.define_metric("morphology/*", step_metric="num_updates")
    wandb.define_metric("accel_actions/*", step_metric="num_updates")
    wandb.define_metric("accel_parameters/*", step_metric="num_updates")
    wandb.define_metric("kangaroo_layout/*", step_metric="num_updates")
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
