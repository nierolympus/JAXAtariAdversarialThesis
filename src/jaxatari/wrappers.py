"""Jaxatari Wrappers"""

import functools
import types
import warnings
from typing import Any, Dict, Tuple, Union, Optional, Callable
from dataclasses import is_dataclass, asdict

import chex
from flax import struct
import jax
import jax.image as jim
import jax.numpy as jnp
from jax import flatten_util
from jaxatari.environment import EnvState, JAXAtariAction as Action
import jaxatari.spaces as spaces
import numpy as np
from jaxatari.rendering.jax_rendering_utils import RendererConfig

class JaxatariWrapper(object):
    """Base class for JAXAtark wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)
    
class MultiRewardWrapper(JaxatariWrapper):
    """
    Allows providing multiple reward functions to be computed at every step.
    Apply this wrapper directly after the base environment, before any other wrappers.
    """

    def __init__(self, env, reward_funcs: list[Callable]):
        super().__init__(env)
        assert isinstance(reward_funcs, list) and len(reward_funcs) > 0, "reward_funcs must be a non-empty list of callables" 
        self._reward_funcs = reward_funcs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: EnvState, state: EnvState) -> chex.Array:
        """Compute multiple rewards based on the provided reward functions."""
        if self._reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self._reward_funcs]
        )
        return rewards

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: int) -> Tuple[chex.Array, EnvState, float, bool, Dict]: 
        obs, new_state, reward, done, info = self._env.step(state, action)
        all_rewards = self._get_all_rewards(state, new_state)
        # Convert info to dict: handle NamedTuple (has _asdict) or dataclass (use asdict)
        if hasattr(info, '_asdict'):
            info = info._asdict()
        elif is_dataclass(info):
            info = asdict(info)
        info["all_rewards"] = all_rewards
        return obs, new_state, reward, done, info 

@struct.dataclass
class AtariState:
    env_state: EnvState
    key: chex.PRNGKey
    step: int
    prev_action: int
    obs_stack: chex.Array
    
class AtariWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the rendered image and object-centric observations unflattened.
    Both are stacked by frame_stack_size.
    Args:
        env: The environment to wrap.
        sticky_actions: Whether to use sticky actions.
        frame_stack_size: The number of frames to stack.
        frame_skip: The number of frames to skip.
    """
    # TODO: change sticky_actions to float
    def __init__(self, env, sticky_actions: bool = True, frame_stack_size: int = 4, frame_skip: int = 4, max_episode_length: int = 10_000, episodic_life: bool = True, first_fire: bool = True, noop_reset: int = 0, clip_reward: bool = False, max_pooling: bool = False, full_action_space: bool = False,):
        super().__init__(env)
        self._env = env
        self.sticky_actions = sticky_actions
        self.frame_stack_size = frame_stack_size
        self.frame_skip = frame_skip
        self.max_episode_length = max_episode_length
        self.episodic_life = episodic_life
        self.first_fire = first_fire
        self.noop_reset = False if noop_reset == 0 else True
        self.noop_max = noop_reset
        self.clip_reward = clip_reward
        self.max_pooling = max_pooling
        self.full_action_space = full_action_space

        # --- 1) HANDLE FULL ACTION SPACE LOGIC ---
        # If requested, swap the environment's (minimal) action set for the full identity set.
        # This keeps each game env "clean" while enabling a central switch for experimentation.
        if self.full_action_space and hasattr(self._env, "ACTION_SET"):
            # Overwrite the instance attribute with [0, 1, ... 17]
            self._env.ACTION_SET = jnp.arange(18, dtype=jnp.int32)

        # --- 2) RESOLVE CORRECT 'FIRE' ACTION INDEX ---
        # The wrapped env expects an *index* into ACTION_SET (agent action), not the ALE action constant.
        self.fire_action_index: int = int(Action.FIRE)  # fallback if env doesn't expose ACTION_SET
        self.first_fire = first_fire

        if hasattr(self._env, "ACTION_SET"):
            # Convert to numpy for search (safe in __init__)
            action_set_np = np.array(self._env.ACTION_SET)
            fire_indices = np.where(action_set_np == int(Action.FIRE))[0]
            if len(fire_indices) > 0:
                self.fire_action_index = int(fire_indices[0])
            else:
                # Game has no FIRE action (e.g. Freeway).
                # Disable first_fire to prevent sending a random command by mistake.
                self.first_fire = False

        self._observation_space = spaces.stack_space(self._env.observation_space(), self.frame_stack_size)

    def observation_space(self) -> spaces.Space:
        """Returns the stacked observation space."""
        return self._observation_space
    
    def image_space(self) -> spaces.Box:
        """Returns the image space."""
        return self._env.image_space()

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        # Split keys for all potential random operations
        env_key, wrapper_key, noop_key = jax.random.split(key, 3)
        obs, env_state = self._env.reset(env_key)
        step = jnp.array(0, dtype=jnp.int32)
        prev_action = jnp.array(0, dtype=jnp.int32)

        # TODO: in which order should the noop and first_fire be done?
        # ========== NOOP RESET ==========
        def perform_noop_reset(carry):
            # This function will be executed if self.noop_reset is True
            env_state, obs, step = carry
            # Generate the random number of no-op steps to take.
            num_noops = jax.random.randint(noop_key, shape=(), minval=0, maxval=self.noop_max + 1)

            def noop_body_fn(i, loop_carry):
                current_env_state, current_obs = loop_carry
                # We always compute the next step for static graph tracing...
                next_obs, next_env_state, _, _, _ = self._env.step(current_env_state, Action.NOOP)
                # ...but only apply the update if the loop index is less than our dynamic random number.
                env_state_out = jax.lax.cond(i < num_noops, lambda: next_env_state, lambda: current_env_state)
                obs_out = jax.lax.cond(i < num_noops, lambda: next_obs, lambda: current_obs)
                return env_state_out, obs_out

            # Loop for the static maximum number of no-ops.
            final_env_state, final_obs = jax.lax.fori_loop(0, self.noop_max, noop_body_fn, (env_state, obs))
            
            # Update the step counter by the dynamic number of no-ops performed.
            final_step = step + num_noops
            return final_env_state, final_obs, final_step

        # Use lax.cond to conditionally apply the whole no-op block based on the static self.noop_reset flag.
        env_state, obs, step = jax.lax.cond(
            self.noop_reset,
            lambda carry: perform_noop_reset(carry),
            lambda carry: carry,
            (env_state, obs, step)
        )

        # ========== FIRST FIRE ==========
        def perform_first_fire(carry):
            env_state, obs, step, _ = carry
            fire_obs, fire_env_state, _, _, _ = self._env.step(env_state, self.fire_action_index)
            return fire_env_state, fire_obs, step + 1, self.fire_action_index

        def identity_fire(carry):
            return carry
        
        # Conditionally apply the fire action based on the static self.first_fire flag.
        env_state, obs, step, prev_action = jax.lax.cond(
            self.first_fire,
            perform_first_fire,
            identity_fire,
            (env_state, obs, step, prev_action)
        )

        # Create the initial frame stack from the final observation.
        obs = jax.tree.map(lambda x: jnp.stack([x] * self.frame_stack_size), obs)

        return obs, AtariState(env_state, wrapper_key, step, prev_action, obs)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: AtariState, action: Union[int, float]) -> Tuple[Tuple[chex.Array, chex.Array], AtariState, float, bool, Dict[Any, Any]]:
        step_key, next_state_key = jax.random.split(state.key)

        new_action = action
        # Use lax.cond and fix shape for scalar actions
        use_sticky_action = jax.random.uniform(step_key, shape=()) < 0.25
        new_action = jax.lax.cond(self.sticky_actions & use_sticky_action, lambda: state.prev_action, lambda: action)

        # use scan to step the env for frame_skip times
        def body_fn(carry, _):
            env_state, action = carry
            obs, new_env_state, reward, done, info = self._env.step(env_state, action) 
            return (new_env_state, action), (obs, reward, done, info)

        (new_env_state, new_action), (obs, rewards, dones, infos) = jax.lax.scan(
            body_fn,
            (state.env_state, new_action),
            None,
            length=self.frame_skip,
        )

        # ========== MAX POOLING LOGIC ==========
        def do_max_pool(obs_pytree):
            # Take the element-wise maximum over the last two frames.
            last_obs = jax.tree.map(lambda x: x[-1], obs_pytree)
            second_last_obs = jax.tree.map(lambda x: x[-2], obs_pytree)
            return jax.tree.map(jnp.maximum, last_obs, second_last_obs)

        def take_last_frame(obs_pytree):
            # Default behavior: just take the final frame.
            return jax.tree.map(lambda x: x[-1], obs_pytree)
        
        # Conditionally apply max-pooling based on the static flag.
        latest_obs = jax.lax.cond(self.max_pooling, do_max_pool, take_last_frame, obs)

        # push latest obs into the stack
        new_obs_stack = jax.tree.map(lambda stack, obs_leaf: jnp.concatenate([stack[1:], jnp.expand_dims(obs_leaf, axis=0)], axis=0), state.obs_stack, latest_obs)

        reward = jnp.sum(rewards)
        real_done = jnp.logical_or(dones.any(), state.step >= self.max_episode_length)
        done = real_done
        if self.episodic_life:
            # If the player has lost a life, we consider the episode done
            if hasattr(state.env_state, "lives"):
                done = jnp.logical_or(done, new_env_state.lives < state.env_state.lives)
            elif hasattr(state.env_state, "lives_lost"):
                done = jnp.logical_or(done, new_env_state.lives_lost > state.env_state.lives_lost)

        def reduce_info(k, v):
            if k == "all_rewards":
                return v.sum(axis=0)
            else:
                return v[-1]

        if hasattr(infos, '_asdict'):
            # It's a namedtuple or similar, convert to dict
            info_items = infos._asdict().items()
        elif is_dataclass(infos):
            # It's a dataclass, convert to dict
            info_items = asdict(infos).items()
        else:
            # It's already a dict
            info_items = infos.items()

        info_dict = {k: reduce_info(k, v) for k, v in info_items}

        # Use jax.lax.cond to correctly handle state and key propagation on reset
        def _reset_fn(_):
            # When done, reset. The new state will contain the properly advanced next_state_key.
            return self.reset(next_state_key)
        
        def _softreset_fn(_):
            # When just done (not real_done, episodic_life) we keep the env_state but reset the step counter
            next_state = AtariState(new_env_state, next_state_key, 0, new_action, new_obs_stack)
            return new_obs_stack, next_state

        def _step_fn(_):
            # When not done, create the next state, passing next_state_key for the *next* step.
            next_state = AtariState(new_env_state, next_state_key, state.step + 1, new_action, new_obs_stack)
            return new_obs_stack, next_state

        #Note: Using real_done here, since we don't want to reset the game if it's just a life lost. (only send done signal)
        new_obs, new_state = jax.lax.cond(
            real_done,
            _reset_fn,
            lambda _: jax.lax.cond(
                done, # done, but not real_done
                _softreset_fn,
                _step_fn, # not done at all
                operand=None
            ),
            operand=None)

        # store actual done - not affected by episodic life
        info_dict["env_done"] = real_done

        # store actual reward in info dict before clipping
        info_dict["env_reward"] = reward
        reward = jax.lax.cond(
            self.clip_reward,
            lambda reward: jnp.sign(reward),
            lambda reward: reward,
            reward
        )

        return new_obs, new_state, reward, done, info_dict


class ObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns stacked object-centric observations.
    The output observation is a 2D array of shape (frame_stack_size, num_features).
    Apply this wrapper after the AtariWrapper!
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "ObjectCentricWrapper must be applied after AtariWrapper"

        # Calculate exact bounds for the flattened observation based on the space structure.
        # Get the stacked observation space from AtariWrapper.
        stacked_space = self._env.observation_space()
        lows, highs = [], []
        
        # Iterate over leaves of the stacked space. Each leaf is a Box(stack_size, ...).
        # We extract bounds for a single frame (index 0) and flatten them.
        for leaf_space in jax.tree.leaves(stacked_space):
            if isinstance(leaf_space, spaces.Box):
                # Extract bounds from the first frame and flatten
                low_arr = np.broadcast_to(leaf_space.low[0], leaf_space.shape[1:]).flatten()
                high_arr = np.broadcast_to(leaf_space.high[0], leaf_space.shape[1:]).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
            else:
                # Should not happen if stack_space works correctly (it converts Discrete to Box)
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")
        
        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        # create the 2D Box space
        self._observation_space = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self._env.frame_stack_size, int(single_frame_lows.shape[0])),
            dtype=jnp.float32
        )
    
    def observation_space(self) -> spaces.Box:
        """Returns a Box space for the flattened observation."""
        return self._observation_space

    @functools.partial(jax.jit, static_argnums=(0,))
    def _flatten_obs(self, obs_stack):
        """Flatten each frame in the observation stack using ravel_pytree."""
        flattened = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])(obs_stack)
        return flattened.astype(jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        obs, state = self._env.reset(key)
        # Flatten each frame in the stack
        flat_obs = self._flatten_obs(obs)
        return flat_obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: AtariState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:  # dict]:
        obs, state, reward, done, info = self._env.step(state, action)
        # Flatten each frame in the stack
        flat_obs = self._flatten_obs(obs)
        return flat_obs, state, reward, done, info


@struct.dataclass 
class PixelState:
    atari_state: AtariState
    image_stack: chex.Array


class PixelObsWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations.
    Apply this wrapper after the AtariWrapper!
    """
    # TODO: remove do_pixel_resize and resize whenever a different shape / grayscale is given?
    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False, use_native_downscaling: bool = False):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelObsWrapper has to be applied after AtariWrapper"

        # Access the Base Environment
        base_env = self._env._env if isinstance(self._env, AtariWrapper) else self._env

        if do_pixel_resize and use_native_downscaling:
            # call helper from modifications to make sure that applied mods remain applied after native downscaling (lazy import to avoid circular dependency)
            from jaxatari.modification import apply_native_downscaling
            self.do_pixel_resize, self.grayscale = apply_native_downscaling(
                base_env, pixel_resize_shape, grayscale
            )
            self.pixel_resize_shape = pixel_resize_shape
        else:
            self.do_pixel_resize = do_pixel_resize
            self.pixel_resize_shape = pixel_resize_shape
            self.grayscale = grayscale

        # Dynamically calculate the final observation space shape
        # If we hot-swapped, image_space() will now return the correct small size automatically
        final_shape = self._env.image_space().shape
        
        # If we are doing wrapper-side resizing (legacy), we still calculate manually
        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
            channels = 1 if self.grayscale else final_shape[2]
            final_shape = (height, width, channels)

        # Create the space for a single preprocessed frame
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        # Stack the single-frame space
        self._observation_space = spaces.stack_space(image_space, self._env.frame_stack_size)

    def observation_space(self) -> spaces.Box:
        """Returns the stacked image space."""
        return self._observation_space
    
    def _preprocess_image(self, image: chex.Array) -> chex.Array:
        """Applies resizing and grayscaling to a single image frame."""
        image = image.astype(jnp.float32)

        # Has to use a standard Python `if` since jax.lax.cond would fail due to different shapes. This is possible since do_pixel_resize is a static parameter.
        if self.do_pixel_resize:
            image = jim.resize(image, (self.pixel_resize_shape[0], self.pixel_resize_shape[1], image.shape[-1]), method='bilinear')
        
        # applies grayscale if enabled with the same method as for resize
        if self.grayscale:
            image = jnp.dot(image, jnp.array([0.2989, 0.5870, 0.1140]))[..., jnp.newaxis] # numbers for grayscale transformation as in https://en.wikipedia.org/wiki/Luma_(video)
        
        return image.astype(jnp.uint8)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, PixelState]:
        # The underlying AtariWrapper returns its own state, which we store.
        _, atari_state = self._env.reset(key)
        image = self._env.render(atari_state.env_state)
        
        processed_image = self._preprocess_image(image)

        # Create a stack of identical processed images for the initial state
        image_stack = jnp.stack([processed_image] * self._env.frame_stack_size)
        
        return image_stack, PixelState(atari_state, image_stack)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # Pass the nested atari_state to the underlying wrapper's step function
        _, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)

        # Update the image stack by shifting and adding the new processed image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)

        # Create the new state with the *new* atari_state from the step
        new_state = PixelState(atari_state, image_stack)
        return image_stack, new_state, reward, done, info

@struct.dataclass 
class PixelAndObjectCentricState:
    atari_state: AtariState
    image_stack: chex.Array
    obs_stack: chex.Array

class PixelAndObjectCentricWrapper(JaxatariWrapper):
    """
    Wrapper for Atari environments that returns the flattened pixel observations and object-centric observations.
    Apply this wrapper after the AtariWrapper!
    """
    
    def __init__(self, env, do_pixel_resize: bool = False, pixel_resize_shape: tuple[int, int] = (84, 84), grayscale: bool = False, use_native_downscaling: bool = False):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "PixelAndObjectCentricWrapper must be applied after AtariWrapper"
        
        # Access the Base Environment
        base_env = self._env._env if isinstance(self._env, AtariWrapper) else self._env

        if do_pixel_resize and use_native_downscaling:
            # call helper from modifications to make sure that applied mods remain applied after native downscaling (lazy import to avoid circular dependency)
            from jaxatari.modification import apply_native_downscaling
            self.do_pixel_resize, self.grayscale = apply_native_downscaling(
                base_env, pixel_resize_shape, grayscale
            )
            self.pixel_resize_shape = pixel_resize_shape
        else:
            self.do_pixel_resize = do_pixel_resize
            self.pixel_resize_shape = pixel_resize_shape
            self.grayscale = grayscale

        # Part 1: Define the stacked image space.
        # If we hot-swapped, image_space() will now return the correct small size automatically
        final_shape = self._env.image_space().shape
        
        # If we are doing wrapper-side resizing (legacy), we still calculate manually
        if self.do_pixel_resize:
            height, width = self.pixel_resize_shape
            channels = 1 if self.grayscale else final_shape[2]
            final_shape = (height, width, channels)
        
        image_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=jnp.uint8)
        stacked_image_space = spaces.stack_space(image_space, self._env.frame_stack_size)

        # Part 2: Define the FLATTENED object space with exact bounds.
        # Get the stacked observation space from AtariWrapper.
        stacked_space = self._env.observation_space()
        lows, highs = [], []
        
        # Iterate over leaves of the stacked space. Each leaf is a Box(stack_size, ...).
        # We extract bounds for a single frame (index 0) and flatten them.
        for leaf_space in jax.tree.leaves(stacked_space):
            if isinstance(leaf_space, spaces.Box):
                # Extract bounds from the first frame and flatten
                low_arr = np.broadcast_to(leaf_space.low[0], leaf_space.shape[1:]).flatten()
                high_arr = np.broadcast_to(leaf_space.high[0], leaf_space.shape[1:]).flatten()
                lows.append(low_arr)
                highs.append(high_arr)
            else:
                # Should not happen if stack_space works correctly (it converts Discrete to Box)
                raise TypeError(f"Unsupported space type for flattening: {type(leaf_space)}")
        
        if not lows:
            raise ValueError("The observation space appears to be empty or contain unsupported types.")

        single_frame_lows = np.concatenate(lows)
        single_frame_highs = np.concatenate(highs)

        stacked_object_space_flat = spaces.Box(
            low=single_frame_lows,
            high=single_frame_highs,
            shape=(self._env.frame_stack_size, int(single_frame_lows.shape[0])),
            dtype=jnp.float32
        )

        # Part 3: Combine them into the final Tuple space.
        self._observation_space = spaces.Tuple((
            stacked_image_space,
            stacked_object_space_flat
        ))
    
    
    def observation_space(self) -> spaces.Tuple:
        """Returns a Tuple space containing stacked image and object spaces."""
        return self._observation_space
    
    def _preprocess_image(self, image: chex.Array) -> chex.Array:
        """Applies resizing and grayscaling to a single image frame."""
        image = image.astype(jnp.float32)

        # Has to use a standard Python `if` since jax.lax.cond would fail due to different shapes. This is possible since do_pixel_resize is a static parameter.
        if self.do_pixel_resize:
            image = jim.resize(image, (self.pixel_resize_shape[0], self.pixel_resize_shape[1], image.shape[-1]), method='bilinear')
        
        # applies grayscale if enabled with the same method as for resize
        if self.grayscale:
            image = jnp.dot(image, jnp.array([0.2989, 0.5870, 0.1140]))[..., jnp.newaxis] # numbers for grayscale transformation as in https://en.wikipedia.org/wiki/Luma_(video)
        
        return image.astype(jnp.uint8)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _flatten_obs(self, obs_stack):
        """Flatten each frame in the observation stack using ravel_pytree."""
        return jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])(obs_stack).astype(jnp.float32)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        # 1. Get the initial object observation stack and state from the AtariWrapper
        obs_stack, atari_state = self._env.reset(key)

        # 2. Flatten the object-centric part
        flat_obs = self._flatten_obs(obs_stack)

        # 3. Render and preprocess the image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        image_stack = jnp.stack([processed_image] * self._env.frame_stack_size)

        # 4. Create the state and observation tuple
        new_state = PixelAndObjectCentricState(atari_state, image_stack, flat_obs)
        return (image_stack, flat_obs), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # 1. Step the underlying environment using its state
        obs_stack, atari_state, reward, done, info = self._env.step(state.atari_state, action)

        # 2. Flatten the new object-centric observation stack
        flat_obs = self._flatten_obs(obs_stack)

        # 3. Render and preprocess the new image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        
        # 4. Update the image stack with the new processed image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)
        
        # 5. Create the new state with the new atari_state
        new_state = PixelAndObjectCentricState(atari_state, image_stack, flat_obs)
        return (image_stack, flat_obs), new_state, reward, done, info
    
class PixelAndObjectObsWrapper(PixelAndObjectCentricWrapper):
    """
    Exactly the same as PixelAndObjectCentricWrapper, but return structured OC-obs instead of flattened array.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        # 1. Get the initial object observation stack and state from the AtariWrapper
        obs_stack, atari_state = self._env.reset(key)

        # 3. Render and preprocess the image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        image_stack = jnp.stack([processed_image] * self._env.frame_stack_size)

        # 4. Create the state and observation tuple
        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: PixelAndObjectCentricState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:
        # 1. Step the underlying environment using its state
        obs_stack, atari_state, reward, done, info = self._env.step(state.atari_state, action)

        # 3. Render and preprocess the new image
        image = self._env.render(atari_state.env_state)
        processed_image = self._preprocess_image(image)
        
        # 4. Update the image stack with the new processed image
        image_stack = jnp.concatenate([state.image_stack[1:], jnp.expand_dims(processed_image, axis=0)], axis=0)
        
        # 5. Create the new state with the new atari_state
        new_state = PixelAndObjectCentricState(atari_state, image_stack, obs_stack)
        return (image_stack, obs_stack), new_state, reward, done, info


class FlattenObservationWrapper(JaxatariWrapper):
    """
    A wrapper that flattens each leaf array in an observation Pytree.

    Compatible with all the other wrappers, flattens the observations whilst preserving the overarching structure (i.e. if the observation is a tuple of multiple observations, the flattened observation will be a tuple of flattened observations).
    """

    def __init__(self, env):
        super().__init__(env)

        # build the new (flattened) observation space
        original_space = self._env.observation_space()

        def flatten_space(space: spaces.Box) -> spaces.Box:
            # Create flattened low/high arrays by broadcasting the original bounds
            # and then reshaping. This preserves the bounds for each element.
            flat_low = np.broadcast_to(space.low, space.shape).flatten()
            flat_high = np.broadcast_to(space.high, space.shape).flatten()
            
            return spaces.Box(
                low=jnp.array(flat_low),
                high=jnp.array(flat_high),
                dtype=space.dtype
            )
        
        self._observation_space = jax.tree.map(
            flatten_space,
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

    def observation_space(self) -> spaces.Space:
        """Returns a space where each leaf array is flattened."""
        return self._observation_space

    def _process_obs(self, obs_tree: chex.ArrayTree) -> chex.ArrayTree:
        """Applies .flatten() to each leaf array in the pytree."""
        def flatten_and_cast(leaf):
            flattened = leaf.flatten()
            # Cast to float32 to match space dtype
            return flattened.astype(jnp.float32) if isinstance(leaf, jnp.ndarray) else flattened
        return jax.tree.map(flatten_and_cast, obs_tree)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        processed_obs = self._process_obs(obs)
        return processed_obs, state # State can be passed through directly

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, Any, float, bool, Dict[str, Any]]:
        obs, next_state, reward, done, info = self._env.step(state, action)
        processed_obs = self._process_obs(obs)
        return processed_obs, next_state, reward, done, info
    

class NormalizeObservationWrapper(JaxatariWrapper):
    """
    A wrapper that normalizes each leaf in an observation Pytree.
    This wrapper is compatible with any observation structure (Pytrees).
    """

    def __init__(self, env, to_neg_one: bool = False, dtype=jnp.float16):
        super().__init__(env)
        self._to_neg_one = to_neg_one
        self._dtype = dtype

        original_space = self._env.observation_space()

        # Create Pytrees of the same structure as observations, but holding the low/high bounds.
        self._low = jax.tree.map(
            lambda s: jnp.array(s.low, dtype=s.dtype),
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )
        self._high = jax.tree.map(
            lambda s: jnp.array(s.high, dtype=s.dtype),
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

        # The new observation space will have the same structure, but all leaves
        def _normalize_space(space: spaces.Box) -> spaces.Box:
            low_val = -1.0 if self._to_neg_one else 0.0
            return spaces.Box(
                low=low_val,
                high=1.0,
                shape=space.shape,
                dtype=self._dtype
            )

        self._observation_space = jax.tree.map(
            _normalize_space,
            original_space,
            is_leaf=lambda x: isinstance(x, spaces.Box)
        )

    def observation_space(self) -> spaces.Space:
        """Returns the normalized observation space where leaves are in [0, 1]."""
        return self._observation_space

    def _normalize_leaf(self, obs_leaf, low_leaf, high_leaf):
        """Helper function to normalize a single leaf array."""
        obs_leaf = obs_leaf.astype(self._dtype)
        
        # Calculate the range and scale for normalization
        range_leaf = high_leaf.astype(self._dtype) - low_leaf.astype(self._dtype)
        scale = 1.0 / jnp.where(range_leaf > 1e-8, range_leaf, 1.0)
        
        # Normalize to [0, 1]
        normalized_0_1 = (obs_leaf - low_leaf.astype(self._dtype)) * scale

        # Conditionally shift to [-1, 1]
        final_normalized = jax.lax.cond(
            self._to_neg_one,
            lambda x: 2.0 * x - 1.0,
            lambda x: x,
            normalized_0_1
        )
        
        # Clip to ensure values are within the target range
        clip_low = -1.0 if self._to_neg_one else 0.0
        return jnp.clip(final_normalized, clip_low, 1.0)

    def _normalize_obs(self, obs: chex.ArrayTree) -> chex.ArrayTree:
        """
        Applies normalization to each leaf array in the observation pytree,
        robustly handling structural mismatches between observation and space Pytrees.
        """
        # Get the leaves of all pytrees. Since the number of leaves and their
        # order is guaranteed to be the same, we can work with the flat lists.
        obs_leaves = jax.tree.leaves(obs)
        low_leaves = jax.tree.leaves(self._low)
        high_leaves = jax.tree.leaves(self._high)

        # Apply the normalization to each corresponding leaf triplet.
        normalized_leaves = [
            self._normalize_leaf(o, l, h)
            for o, l, h in zip(obs_leaves, low_leaves, high_leaves)
        ]

        # Reconstruct the output pytree with the same structure as the input 'obs'.
        return jax.tree.unflatten(jax.tree.structure(obs), normalized_leaves)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.ArrayTree, Any]:
        obs, state = self._env.reset(key)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: Any,
        action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, Any, float, bool, Dict[str, Any]]:
        obs, next_state, reward, done, info = self._env.step(state, action)
        normalized_obs = self._normalize_obs(obs)
        return normalized_obs, next_state, reward, done, info



@struct.dataclass
class LogState:
    atari_state: Any # Can be any of the states from wrappers above
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

class LogWrapper(JaxatariWrapper):
    """Log episode returns and lengths. An episode ends when the wrapped env returns done=True.
    Uses env_reward from info when present (unclipped); otherwise uses the step reward.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, LogState]:
        obs, atari_state = self._env.reset(key)
        state = LogState(atari_state, 0.0, 0, 0.0, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: LogState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogState, float, bool, Dict[Any, Any]]:
        obs, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        # use env_reward (unclipped) for logging when available
        new_episode_return = state.episode_returns + info.get("env_reward", reward)
        new_episode_length = state.episode_lengths + 1
        done_ = jnp.bool_(done)
        state = LogState(
            atari_state=atari_state,
            episode_returns=jnp.where(done_, jnp.float32(0), jnp.float32(new_episode_return)),
            episode_lengths=jnp.where(done_, jnp.int32(0), jnp.int32(new_episode_length)),
            returned_episode_returns=jnp.where(
                done_, jnp.float32(new_episode_return), jnp.float32(state.returned_episode_returns)
            ),
            returned_episode_lengths=jnp.where(
                done_, jnp.int32(new_episode_length), jnp.int32(state.returned_episode_lengths)
            ),
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info

@struct.dataclass
class MultiRewardLogState:
    atari_state: Any # Can be any of the states from wrappers above
    episode_returns_env: float
    episode_returns: chex.Array
    episode_lengths: int
    returned_episode_returns_env: float
    returned_episode_returns: chex.Array
    returned_episode_lengths: int

class MultiRewardLogWrapper(JaxatariWrapper):
    """Log episode returns and lengths for multiple rewards. An episode ends when the wrapped env returns done=True.
    Apply MultiRewardWrapper to the core env when using this wrapper.
    Final logs: 'returned_episode_returns_0', ... for each reward function; env reward in 'returned_episode_env_returns'.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey,
    ) -> Tuple[chex.Array, MultiRewardLogState]:
        obs, atari_state = self._env.reset(key)
        # Dummy step to get info structure 
        _, _, _, _, dummy_info = self._env.step(atari_state, 0)
        rewards_shape_provider = dummy_info.get("all_rewards", jnp.zeros(1))
        episode_returns_init = jnp.zeros_like(rewards_shape_provider)
        state = MultiRewardLogState(atari_state, 0.0, episode_returns_init, 0, 0.0, episode_returns_init, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: MultiRewardLogState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, MultiRewardLogState, float, bool, Dict[Any, Any]]:
        obs, atari_state, reward, done, info = self._env.step(state.atari_state, action)
        new_episode_return_env = state.episode_returns_env + info.get("env_reward", reward)
        all_rewards_step = info.get("all_rewards", jnp.zeros_like(state.episode_returns))
        new_episode_return = state.episode_returns + all_rewards_step
        new_episode_length = state.episode_lengths + 1
        done_ = jnp.bool_(done)
        state = MultiRewardLogState(
            atari_state=atari_state,
            episode_returns_env=jnp.where(done_, jnp.float32(0), jnp.float32(new_episode_return_env)),
            episode_returns=jnp.where(done_, jnp.zeros_like(state.episode_returns), new_episode_return),
            episode_lengths=jnp.where(done_, jnp.int32(0), jnp.int32(new_episode_length)),
            returned_episode_returns_env=jnp.where(
                done_, jnp.float32(new_episode_return_env), jnp.float32(state.returned_episode_returns_env)
            ),
            returned_episode_returns=jnp.where(
                done_, new_episode_return, state.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                done_, jnp.int32(new_episode_length), jnp.int32(state.returned_episode_lengths)
            ),
        )
        info["returned_episode_env_returns"] = state.returned_episode_returns_env
        for i, r in enumerate(new_episode_return):
            info[f"returned_episode_returns_{i}"] = state.returned_episode_returns[i]
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info
@dataclass(frozen=True)
class GenericModSpec:
    """Declarative definition of a single state modification."""

    target: str
    op: str = "add"  # add|sub|decrease|increase|mul|scale|set
    value: float = 0.0
    const: Optional[str] = None
    factor: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_const: Optional[str] = None
    max_const: Optional[str] = None
    preserve_sign: bool = False
    dtype: Any = jnp.int32


def _unwrap_env_chain(env: Any) -> Any:
    """Returns the innermost env by following `_env` links."""
    core_env = env
    while hasattr(core_env, "_env"):
        core_env = core_env._env
    return core_env


def _to_plain_mapping(value: Any) -> Any:
    """Recursively convert constants objects into plain dict/list/scalars."""
    if isinstance(value, dict):
        return {k: _to_plain_mapping(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_mapping(v) for v in value]
    if hasattr(value, "_asdict"):
        return {k: _to_plain_mapping(v) for k, v in value._asdict().items()}
    if is_dataclass(value):
        return {k: _to_plain_mapping(v) for k, v in asdict(value).items()}
    if hasattr(value, "__dict__"):
        return {k: _to_plain_mapping(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


class GenericStateModWrapper(JaxatariWrapper):
    """
    Generic adversarial wrapper for state modifications via declarative mod specs.
    Supports a secondary mod action in `step(..., adversary_action=...)` or legacy dict action API.
    """

    def __init__(
            self,
            env,
            mod_specs: list[GenericModSpec],
            info_fields: Optional[Dict[str, str]] = None,
            constant_overrides: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env)
        self._action_space = self._env.action_space()
        self._core_env = _unwrap_env_chain(self._env)
        self._mod_specs = mod_specs
        self._info_fields = {
            key: tuple(path.split(".")) for key, path in (info_fields or {}).items()
        }

        if constant_overrides:
            self.set_constant_overrides(constant_overrides)

        self._mod_fns = self._build_mod_fns()

    def action_space(self) -> spaces.Space:
        return self._action_space

    def get_constants_dict(self) -> Dict[str, Any]:
        """Returns a recursively unpacked constants dictionary."""
        return _to_plain_mapping(self._core_env.consts)

    def _resolve_path(self, obj: Any, path_parts: tuple[str, ...]) -> Any:
        current = obj
        for part in path_parts:
            if isinstance(current, dict):
                current = current[part]
            else:
                current = getattr(current, part)
        return current

    def _set_path(self, obj: Any, path_parts: tuple[str, ...], value: Any) -> Any:
        head = path_parts[0]
        if len(path_parts) == 1:
            if isinstance(obj, dict):
                out = dict(obj)
                out[head] = value
                return out
            if hasattr(obj, "replace"):
                return obj.replace(**{head: value})
            if hasattr(obj, "_replace"):
                return obj._replace(**{head: value})
            setattr(obj, head, value)
            return obj

        child = obj[head] if isinstance(obj, dict) else getattr(obj, head)
        new_child = self._set_path(child, path_parts[1:], value)

        if isinstance(obj, dict):
            out = dict(obj)
            out[head] = new_child
            return out
        if hasattr(obj, "replace"):
            return obj.replace(**{head: new_child})
        if hasattr(obj, "_replace"):
            return obj._replace(**{head: new_child})
        setattr(obj, head, new_child)
        return obj

    def _resolve_const(self, path: str) -> Any:
        return self._resolve_path(self._core_env.consts, tuple(path.split(".")))

    def set_constant_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Applies constant overrides to the core env constants.
        Path-based keys are supported (e.g. "LEVEL_1.some_value" if the type supports replace/_replace).
        """
        new_consts = self._core_env.consts
        for key, value in overrides.items():
            new_consts = self._set_path(new_consts, tuple(key.split(".")), value)
        self._core_env.consts = new_consts

    def _resolve_amount(self, spec: GenericModSpec) -> chex.Array:
        amount = spec.value
        if spec.const is not None:
            amount = amount + self._resolve_const(spec.const)
        amount = amount * spec.factor
        return jnp.asarray(amount)

    def _resolve_bound(self, literal: Optional[float], const_path: Optional[str]) -> Optional[chex.Array]:
        if const_path is not None:
            return jnp.asarray(self._resolve_const(const_path))
        if literal is not None:
            return jnp.asarray(literal)
        return None

    def _compile_spec_fn(self, spec: GenericModSpec):
        target_path = tuple(spec.target.split("."))
        amount = self._resolve_amount(spec)
        bound_min = self._resolve_bound(spec.min_value, spec.min_const)
        bound_max = self._resolve_bound(spec.max_value, spec.max_const)
        op = spec.op.lower()

        def _apply(s):
            current = self._resolve_path(s, target_path)
            if op in ("add", "increase"):
                new_value = current + amount
            elif op in ("sub", "decrease"):
                new_value = current - amount
            elif op in ("mul", "scale"):
                new_value = current * amount
            elif op == "set":
                new_value = jnp.broadcast_to(amount, current.shape if hasattr(current, "shape") else ())
            else:
                raise ValueError(f"Unsupported op '{spec.op}' in GenericModSpec")

            if spec.preserve_sign and (bound_min is not None or bound_max is not None):
                sign = jnp.where(new_value < 0, -1, 1)
                magnitude = jnp.abs(new_value)
                if bound_min is not None:
                    magnitude = jnp.maximum(magnitude, bound_min)
                if bound_max is not None:
                    magnitude = jnp.minimum(magnitude, bound_max)
                new_value = sign * magnitude
            else:
                if bound_min is not None:
                    new_value = jnp.maximum(new_value, bound_min)
                if bound_max is not None:
                    new_value = jnp.minimum(new_value, bound_max)

            if spec.dtype is not None:
                new_value = new_value.astype(spec.dtype)

            return self._set_path(s, target_path, new_value)

        return _apply

    def _build_mod_fns(self):
        fns = [lambda s: s]  # mod_action=0 => no-op
        fns.extend(self._compile_spec_fn(spec) for spec in self._mod_specs)
        return fns

    @functools.partial(jax.jit, static_argnums=(0,))
    def _apply_mod(self, state, mod_action):
        clipped_mod_action = jnp.clip(mod_action, 0, len(self._mod_fns) - 1)
        return jax.lax.switch(clipped_mod_action, self._mod_fns, state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, adversary_action=None):
        # Support both new API (separate adversary action) and legacy dict API.
        if isinstance(action, dict):
            base_action = action["agent"]
            mod_action = action.get("adversary", 0)
        else:
            base_action = action
            mod_action = 0 if adversary_action is None else adversary_action

        mod_action = jnp.asarray(mod_action, dtype=jnp.int32)

        obs, next_state, reward, done, info = self._env.step(state, base_action)
        next_state = self._apply_mod(next_state, mod_action)

        if hasattr(info, "_asdict"):
            info_dict = info._asdict()
        elif is_dataclass(info):
            info_dict = asdict(info)
        else:
            info_dict = dict(info)

        info_dict["mod_action"] = mod_action
        for key, path_parts in self._info_fields.items():
            info_dict[key] = self._resolve_path(next_state, path_parts)

        if hasattr(self._env, "_get_observation"):
            obs = self._env._get_observation(next_state)

        return obs, next_state, reward, done, info_dict


class PongStateModWrapper(GenericStateModWrapper):
    """
    Example of GenericStateModWrapper in Pong
    """

    def __init__(self, env):
        # Resolve limits from Pong constants once to build stable mod specs.
        core_env = _unwrap_env_chain(env)
        max_acc = len(core_env.consts.PLAYER_ACCELERATION) - 1

        specs = [
            GenericModSpec(target="player_speed", op="decrease", value=1, min_const="MAX_SPEED", max_const="MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
            GenericModSpec(target="player_speed", op="add", value=1, min_const="MAX_SPEED", max_const="MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
            GenericModSpec(target="enemy_speed", op="decrease", value=1, min_const="MAX_SPEED", max_const="MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
            GenericModSpec(target="enemy_speed", op="add", value=1, min_const="MAX_SPEED", max_const="MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
            GenericModSpec(target="acceleration_counter", op="decrease", value=1, min_value=0, max_value=max_acc, dtype=jnp.int32),
            GenericModSpec(target="acceleration_counter", op="add", value=1, min_value=0, max_value=max_acc, dtype=jnp.int32),
            GenericModSpec(target="ball_vel_x", op="decrease", value=1, min_const="MIN_BALL_SPEED", max_const="BALL_MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
            GenericModSpec(target="ball_vel_x", op="add", value=1, min_const="MIN_BALL_SPEED", max_const="BALL_MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
            GenericModSpec(target="ball_vel_y", op="decrease", value=1, min_const="MIN_BALL_SPEED", max_const="BALL_MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
            GenericModSpec(target="ball_vel_y", op="add", value=1, min_const="MIN_BALL_SPEED", max_const="BALL_MAX_SPEED", preserve_sign=True, dtype=jnp.int32),
        ]

        super().__init__(
            env,
            mod_specs=specs,
            info_fields={
                "player_speed": "player_speed",
                "enemy_speed": "enemy_speed",
                "acceleration_counter": "acceleration_counter",
                "ball_vel_x": "ball_vel_x",
                "ball_vel_y": "ball_vel_y",
            },
        )