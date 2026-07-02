"""
PQN + ACCEL curriculum runner.

The protagonist is the same PQN network/train-state used by
`pqn_agent_adv_random.py`. ACCEL stays outside JAX as a host-side level
buffer, while each rollout/learner round is jitted.
"""

import copy
import os
import time
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import jaxatari
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jaxatari.wrappers import (
    AccelDeltaModWrapper,
    AtariWrapper,
    FlattenObservationWrapper,
    GenericModSpec,
    LogWrapper,
    NormalizeObservationWrapper,
    ObjectCentricWrapper,
    PixelObsWrapper,
    PongStateModWrapper,
)
from omegaconf import OmegaConf

from ACCEL_buffer import ACCELLevelBuffer, level_to_mask
from pqn_agent_adv_random import QNetwork, Transition
from train_utils import save_params


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class PongAccelModWrapper(AccelDeltaModWrapper):
    """Delta-amplifying ACCEL wrapper with the same Pong targets as the adversary wrapper."""

    def __init__(self, env):
        specs = [
            GenericModSpec(
                target="ball_vel_x",
                op="mul",
                value=10,
                min_const="MIN_BALL_SPEED",
                max_const="MAX_BALL_SPEED",
                preserve_sign=True,
                dtype=jnp.float32,
            ),
            GenericModSpec(
                target="ball_vel_y",
                op="mul",
                value=1,
                min_const="MIN_BALL_SPEED",
                max_const="MAX_BALL_SPEED",
                preserve_sign=True,
                dtype=jnp.float32,
            ),
        ]
        super().__init__(env, mod_specs=specs)


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


def make_train_accel(config):
    config = copy.deepcopy(config)
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = int(
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0

    train_mods = config.get("TRAIN_MODS", None)
    train_mods_list = train_mods if isinstance(train_mods, list) else [train_mods] if train_mods else None

    env = jaxatari.make(config["ENV_NAME"].lower(), mods=train_mods_list)
    adv_enabled = bool(config.get("ADV_RANDOM", False))
    adv_wrapper = config.get("ADV_WRAPPER", None)
    use_accel = bool(config.get("ADV_ACCEL", False))

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

    def apply_adversary_wrapper(wrapped):
        if not adv_enabled or not adv_wrapper:
            return wrapped
        if adv_wrapper.lower() != "pong":
            raise ValueError(f"Unsupported ADV_WRAPPER: {adv_wrapper}")
        return PongAccelModWrapper(wrapped) if use_accel else PongStateModWrapper(wrapped)

    env = apply_adversary_wrapper(apply_wrappers(env))

    if use_accel:
        if not hasattr(env, "_mod_specs"):
            raise ValueError("ACCEL wrapper missing _mod_specs")
        mod_action_space_n = len(env._mod_specs)
    elif adv_enabled:
        if not hasattr(env, "_mod_fns"):
            raise ValueError("Adversarial wrapper missing _mod_fns")
        mod_action_space_n = len(env._mod_fns)
    else:
        mod_action_space_n = 1

    level_buffer = None
    if use_accel:
        level_buffer = ACCELLevelBuffer(
            capacity=int(config.get("ACCEL_BUFFER_CAPACITY", 64)),
            mod_action_space_n=mod_action_space_n,
            max_level_size=int(config.get("ACCEL_MAX_LEVEL_SIZE", 3)),
            rng=np.random.default_rng(int(config["SEED"])),
            replacement_margin=float(config.get("ACCEL_REPLACEMENT_MARGIN", 0.0)),
            conflict_groups=config.get("ACCEL_CONFLICT_GROUPS", None),
        )
        level_buffer.seed_random(int(config.get("ACCEL_SEED_LEVELS", 16)))

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

    network = QNetwork(
        action_dim=env.action_space().n,
        hidden_size=config.get("HIDDEN_SIZE", 128),
        num_layers=config.get("NUM_LAYERS", 2),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        object_centric=config.get("OBJECT_CENTRIC", True),
    )

    def create_agent(rng):
        init_x = jnp.zeros((1, *env.observation_space().shape))
        network_variables = network.init(rng, init_x, train=False)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )
        return CustomTrainState.create(
            apply_fn=network.apply,
            params=network_variables["params"],
            batch_stats=network_variables["batch_stats"],
            tx=tx,
        )

    def batched_step(env_state, action, rng, step_idx, level_mask, strength):
        if use_accel:
            return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                env_state,
                action,
                level_mask,
                strength,
            )
        if adv_enabled:
            rng, rng_mask = jax.random.split(rng)
            mod_action = jax.random.randint(
                rng,
                shape=action.shape,
                minval=0,
                maxval=mod_action_space_n,
            )
            adv_mode = config.get("ADV_MODE", "per_step")
            if adv_mode == "prob":
                apply_mask = jax.random.bernoulli(
                    rng_mask,
                    float(config.get("ADV_PROB", 1.0)),
                    shape=action.shape,
                )
                mod_action = jnp.where(apply_mask, mod_action, 0)
            elif adv_mode == "every_n":
                apply_step = (step_idx % int(config.get("ADV_EVERY_N", 1))) == 0
                mod_action = jnp.where(apply_step, mod_action, 0)
            return jax.vmap(env.step)(env_state, action, mod_action)
        return jax.vmap(env.step)(env_state, action)

    @jax.jit
    def train_round(train_state, expl_state, rng, level_mask, strength):
        def _step_env(carry, step_idx):
            last_obs, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
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
            new_obs, new_env_state, reward, terminated, truncated, info = batched_step(
                env_state,
                action,
                rng_s,
                step_idx,
                level_mask,
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
            (*expl_state, rng),
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
            targets = jax.tree_util.tree_map(
                lambda x: preprocess_transition(x, _rng),
                lambda_targets,
            )

            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_phase,
                (train_state, rng),
                (minibatches, targets),
            )
            return (train_state, rng), (loss, qvals)

        (train_state, rng), (loss, qvals) = jax.lax.scan(
            _learn_epoch,
            (train_state, rng),
            None,
            config["NUM_EPOCHS"],
        )
        train_state = train_state.replace(n_updates=train_state.n_updates + 1)

        chosen_q = jnp.take_along_axis(
            transitions.q_val,
            transitions.action[..., None],
            axis=-1,
        ).squeeze(-1)
        round_metrics = {
            "grad_steps": train_state.grad_steps,
            "td_loss": loss.mean(),
            "qvals": qvals.mean(),
            **_mean_infos(infos, skip=("level_mask", "strength")),
        }
        return (
            train_state,
            (last_obs, final_env_state),
            rng,
            chosen_q,
            lambda_targets,
            transitions.done,
            round_metrics,
        )

    def _mask_for_level(level):
        mask = level_to_mask(level, mod_action_space_n)
        return jnp.broadcast_to(
            jnp.asarray(mask, dtype=bool),
            (config["NUM_ENVS"], mod_action_space_n),
        )

    def train(rng):
        rng, agent_rng = jax.random.split(rng)
        train_state = create_agent(agent_rng)

        rng, reset_rng = jax.random.split(rng)
        init_obs, env_state = jax.vmap(env.reset)(
            jax.random.split(reset_rng, config["NUM_ENVS"])
        )
        expl_state = (init_obs, env_state)

        metrics_history = []
        dummy_mask = jnp.zeros((config["NUM_ENVS"], mod_action_space_n), dtype=bool)
        strength = jnp.asarray(config.get("ACCEL_STRENGTH", 1.0), dtype=jnp.float32)

        for _ in range(config["NUM_UPDATES"]):
            if use_accel:
                parent_entry = level_buffer.sample(
                    temperature=float(config.get("ACCEL_SAMPLE_TEMPERATURE", 1.0))
                )
                parent_mask = _mask_for_level(parent_entry.level)
                rng, round_rng = jax.random.split(rng)
                (
                    train_state,
                    expl_state,
                    rng,
                    chosen_q,
                    lambda_targets,
                    dones,
                    round_metrics,
                ) = train_round(train_state, expl_state, round_rng, parent_mask, strength)
                parent_regret = float(
                    jax.device_get(jnp.mean(compute_per_env_regret(chosen_q, lambda_targets, dones)))
                )
                level_buffer.maybe_insert(parent_entry.level, parent_regret)

                child_level = level_buffer.mutate(parent_entry.level)
                child_mask = _mask_for_level(child_level)
                rng, round_rng = jax.random.split(rng)
                (
                    train_state,
                    expl_state,
                    rng,
                    chosen_q,
                    lambda_targets,
                    dones,
                    round_metrics,
                ) = train_round(train_state, expl_state, round_rng, child_mask, strength)
                child_regret = float(
                    jax.device_get(jnp.mean(compute_per_env_regret(chosen_q, lambda_targets, dones)))
                )
                level_buffer.maybe_insert(child_level, child_regret)

                metrics = {
                    "env_step": int(jax.device_get(train_state.timesteps)),
                    "update_steps": int(jax.device_get(train_state.n_updates)),
                    "accel/parent_regret": parent_regret,
                    "accel/child_regret": child_regret,
                    "accel/parent_level_size": len(parent_entry.level),
                    "accel/child_level_size": len(child_level),
                    **level_buffer.stats(),
                    **_host_metrics(round_metrics),
                }
            else:
                rng, round_rng = jax.random.split(rng)
                (
                    train_state,
                    expl_state,
                    rng,
                    _chosen_q,
                    _lambda_targets,
                    _dones,
                    round_metrics,
                ) = train_round(train_state, expl_state, round_rng, dummy_mask, strength)
                metrics = {
                    "env_step": int(jax.device_get(train_state.timesteps)),
                    "update_steps": int(jax.device_get(train_state.n_updates)),
                    **_host_metrics(round_metrics),
                }

            metrics_history.append(metrics)
            if config["WANDB_MODE"] != "disabled":
                wandb.log(metrics, step=metrics["env_step"])

        return {"runner_state": (train_state, expl_state, rng), "metrics": metrics_history}

    return train


def single_run_accel(config):
    if "alg" in config:
        config = {**config, **config["alg"]}
    env_name = config["ENV_NAME"]
    oc = "oc" if config.get("OBJECT_CENTRIC", False) else "pixel"
    alg_name = config.get("ALG_NAME", "pqn_accel")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[alg_name.upper(), env_name.upper(), "ACCEL", f"jax_{jax.__version__}"],
        name=config.get("NAME", f"{alg_name}_{env_name}"),
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    t0 = time.time()
    outs = []
    for seed_idx, seed_rng in enumerate(rngs):
        seed_config = copy.deepcopy(config)
        seed_config["SEED"] = int(config["SEED"]) + seed_idx
        out = make_train_accel(seed_config)(seed_rng)
        jax.block_until_ready(out["runner_state"][0].timesteps)
        outs.append(out)
    print(f"Total: {time.time() - t0} seconds.")

    if config.get("SAVE_PATH", None) is not None and outs:
        model_state = outs[0]["runner_state"][0]
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
    raise NotImplementedError("ACCEL uses a host-side level buffer; add a non-jitted sweep loop if needed.")


@hydra.main(version_base=None, config_path="config/alg")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run_accel(config)


if __name__ == "__main__":
    main()
