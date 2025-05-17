import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from typing import Callable, Dict, Optional, Union
from imitation.util import util
from imitation.algorithms.bc import (
    BC,
    BatchIteratorWithEpochEndCallback,
    enumerate_batches,
)
from imitation.algorithms import base as algo_base
from imitation.data import rollout as rollout_module
from imitation.data import types
import tqdm
from stable_baselines3.common import vec_env, policies


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        util.weight_initializer(m.weight, std=0.1)
        util.bias_initializer(m.bias, val=0.01)


def generate_random_demonstrations(
    env: gym.Env, num_episodes: int, rng: np.random.Generator
):
    """
    Generate random demonstration transitions by rolling out a uniform-random policy.
    Returns a Transitions object with attributes: observations, actions, next_observations.
    """

    def random_policy_fn(obs, state=None, mask=None):
        batch_size = obs.shape[0]
        actions = np.array([env.action_space.sample() for _ in range(batch_size)])
        return actions, None

    rollouts = rollout_module.rollout(
        random_policy_fn,
        env,  # TODO: resolve that this should be a venv
        rollout_module.make_sample_until(min_timesteps=None, min_episodes=num_episodes),
        rng=rng,
    )
    transitions = rollout_module.flatten_trajectories(rollouts)
    return transitions


class IDMModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, action_dim),
        )
        self.apply(_init_weights)

    def forward(self, state: th.Tensor, next_state: th.Tensor) -> th.Tensor:
        x = th.cat([state, next_state], dim=-1)
        return self.net(x)


class BCO(BC):
    """
    BCO: Behavioral Cloning from Observation without expert actions.
    Accepts expert demonstrations and IDM demonstrations, using data loaders for both.
    """

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        demonstrations,
        idm_demonstrations,
        policy: Optional[policies.ActorCriticPolicy] = None,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        optimizer_cls: Callable[..., th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        **bc_kwargs,
    ):
        # Initialize BC base for expert demos
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            rng=rng,
            policy=policy,
            demonstrations=demonstrations,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            **bc_kwargs,
        )
        # BCO parameters
        self.state_dim = observation_space.shape[0]

        discrete = isinstance(action_space, gym.spaces.Discrete)

        self.action_dim = action_space.n if discrete else action_space.shape[0]
        # Networks
        # TODO: send to device
        self.idm_net = IDMModel(self.state_dim, self.action_dim)
        # Optimizers
        if optimizer_kwargs and "weight_decay" in optimizer_kwargs:
            raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        optimizer_kwargs = optimizer_kwargs or {}
        self.opt_idm = optimizer_cls(self.idm_net.parameters(), **optimizer_kwargs)
        # Loss functions
        if discrete:  # To do: other possibilities?
            self.idm_criterion = nn.CrossEntropyLoss()
        else:
            self.idm_criterion = nn.MSELoss()
        # IDM data loader
        self._idm_data_loader = algo_base.make_data_loader(
            idm_demonstrations,
            self.minibatch_size,
        )

    def train_idm(
        self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        self.idm_net.to(self.policy.device)  # TODO: remove self.policy
        if reset_tensorboard:
            self._bc_logger.reset_tensorboard_steps()

        self._bc_logger.log_epoch(0)

        losses = []

        def _on_epoch_end(epoch_number: int):
            print(sum(losses) / len(losses))
            losses.clear()
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(
                    f"Epoch {epoch_number} {total_num_epochs_str}",
                    pos=1,
                )
            self._bc_logger.log_epoch(epoch_number + 1)
            if on_epoch_end is not None:
                on_epoch_end()

        mini_per_batch = self.batch_size // self.minibatch_size
        n_minibatches = n_batches * mini_per_batch if n_batches is not None else None

        # Initial IDM training
        assert self._idm_data_loader is not None
        idm_batches = BatchIteratorWithEpochEndCallback(
            self._idm_data_loader,
            n_epochs,
            n_minibatches,
            _on_epoch_end,
        )
        batches_with_stats = enumerate_batches(idm_batches)
        tqdm_progress_bar: Optional[tqdm.tqdm] = None

        if progress_bar:
            batches_with_stats = tqdm.tqdm(
                batches_with_stats,
                unit="batch",
                total=n_minibatches,
            )
            tqdm_progress_bar = batches_with_stats

        self.optimizer.zero_grad()
        for (
            batch_num,
            minibatch_size,
            num_samples_so_far,
        ), batch in batches_with_stats:
            obs_tensor: Union[th.Tensor, Dict[str, th.Tensor]]
            # unwraps the observation if it's a dictobs and converts arrays to tensors
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(
                    x, device=self.policy.device
                ),  # TODO: remove self.policy
                types.maybe_unwrap_dictobs(batch["obs"]),
            ).float()  # TODO: maybe match this with network
            next_obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(
                    x, device=self.policy.device
                ),  # TODO: remove self.policy
                types.maybe_unwrap_dictobs(batch["next_obs"]),
            ).float()

            acts = util.safe_to_tensor(batch["acts"], device=self.policy.device)
            pred = self.idm_net(obs_tensor, next_obs_tensor)
            loss = self.idm_criterion(pred, acts)
            losses.append(loss.item())
            self.opt_idm.zero_grad()
            loss.backward()
            self.opt_idm.step()

    def _get_acts(
        self,
        batch: Dict[str, Union[th.Tensor, list, np.ndarray]],
    ):
        obs_tensor: Union[th.Tensor, Dict[str, th.Tensor]]
        next_obs_tensor: Union[th.Tensor, Dict[str, th.Tensor]]
        obs_tensor = types.map_maybe_dict(
            lambda x: util.safe_to_tensor(x, device=self.policy.device),
            types.maybe_unwrap_dictobs(batch["obs"]),
        ).float()  # TODO: maybe match with network
        next_obs_tensor = types.map_maybe_dict(
            lambda x: util.safe_to_tensor(x, device=self.policy.device),
            types.maybe_unwrap_dictobs(batch["next_obs"]),
        ).float()  # TODO: maybe match with network
        inferred_acts_raw = self.idm_net(obs_tensor, next_obs_tensor)
        inferred_acts_raw = util.safe_to_tensor(
            inferred_acts_raw, device=self.policy.device
        )
        inferred_acts = self._convert_idm_net_output(inferred_acts_raw)
        return inferred_acts

    def _convert_idm_net_output(self, acts):
        """Convert predicted actions output by idm_net to the format of the action space."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            return th.argmax(acts, dim=1)
        elif isinstance(self.action_space, gym.spaces.Box):
            return th.clamp(acts, min=-1, max=1)
