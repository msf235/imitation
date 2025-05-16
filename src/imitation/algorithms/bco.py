import os
import time
import numpy as np
import torch
import torch as th  # TODO: resolve duplicate
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import Optional, Callable
from imitation.util import util
from torch.utils.tensorboard import SummaryWriter
from imitation.algorithms.bc import (
    BC,
    RolloutStatsComputer,
    BatchIteratorWithEpochEndCallback,
    enumerate_batches,
)
from imitation.algorithms import base as algo_base
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout as rollout_module
from imitation.data import types
import tqdm
from stable_baselines3.common import policies, torch_layers, utils, vec_env
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)


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
        env,
        rollout_module.make_sample_until(min_timesteps=None, min_episodes=num_episodes),
        rng=rng,
    )
    transitions = rollout_module.flatten_trajectories(rollouts)
    return transitions


class PolicyModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, action_dim),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)


class BCOfromObservation(BC):
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
        policy_network: Optional[PolicyModel] = None,
        continuous: bool = False,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        optimizer_cls: Callable[..., optim.Optimizer] = optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        **bc_kwargs,
    ):
        # Initialize BC base for expert demos
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            rng=rng,
            policy=None,
            demonstrations=demonstrations,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            **bc_kwargs,
        )
        # TODO: delete or overwrite self.policy
        # BCO parameters
        self.continuous = continuous
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0] if continuous else action_space.n
        # Networks
        # TODO: send to device
        self.policy_net = policy_network or PolicyModel(self.state_dim, self.action_dim)
        self.idm_net = IDMModel(self.state_dim, self.action_dim)
        # Optimizers
        if optimizer_kwargs and "weight_decay" in optimizer_kwargs:
            raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        optimizer_kwargs = optimizer_kwargs or {}
        self.opt_policy = optimizer_cls(
            self.policy_net.parameters(), **optimizer_kwargs
        )
        self.opt_idm = optimizer_cls(self.idm_net.parameters(), **optimizer_kwargs)
        # Loss functions
        if continuous:
            self.policy_criterion = nn.MSELoss()
            self.idm_criterion = nn.MSELoss()
        else:
            self.policy_criterion = nn.CrossEntropyLoss()
            self.idm_criterion = nn.CrossEntropyLoss()
        # IDM data loader
        self._idm_data_loader = algo_base.make_data_loader(
            idm_demonstrations,
            self.minibatch_size,
        )
        self._idm_iter = iter(self._idm_data_loader)

        # self.loss_calculator = BehaviorCloningLossCalculator(ent_weight, l2_weight)

    def pre_demonstration(self):
        """
        Fetch next minibatch from the IDM data loader.
        Returns lists of (state, next_state, action).
        """
        try:
            batch = next(self._idm_iter)
        except StopIteration:
            self._idm_iter = iter(self._idm_data_loader)
            batch = next(self._idm_iter)
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        acts = batch["acts"]
        S = [o for o in obs]
        nS = [n for n in next_obs]
        A = [a for a in acts]
        return S, nS, A

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

        compute_rollout_stats = RolloutStatsComputer(
            log_rollouts_venv,
            log_rollouts_n_episodes,
        )

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
            )
            next_obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(
                    x, device=self.policy.device
                ),  # TODO: remove self.policy
                types.maybe_unwrap_dictobs(batch["next_obs"]),
            )

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
        obs_tensor: Union[
            th.Tensor, Dict[str, th.Tensor]
        ]  # TODO: do the same with next_obs_tensor?
        obs_tensor = types.map_maybe_dict(
            lambda x: util.safe_to_tensor(
                x, device=self.policy.device
            ),  # TODO: remove self.policy
            types.maybe_unwrap_dictobs(batch["obs"]),
        )
        next_obs_tensor = types.map_maybe_dict(
            lambda x: util.safe_to_tensor(
                x, device=self.policy.device
            ),  # TODO: remove self.policy
            types.maybe_unwrap_dictobs(batch["next_obs"]),
        )
        inferred_acts_raw = self.idm_net(obs_tensor, next_obs_tensor)
        inferred_acts_raw = util.safe_to_tensor(
            inferred_acts_raw, device=self.policy.device
        )  # TODO: remove policy reference
        inferred_acts = torch.argmax(inferred_acts_raw, dim=1)
        return inferred_acts

    def update_policy(self, states, actions):
        self.policy_net.train()
        for idx in util.get_shuffle_idx(len(states), self.batch_size):
            bs_np = np.array([states[i] for i in idx], dtype=np.float32)
            bs = torch.from_numpy(bs_np).to(self.policy_net.net[0].weight.device)
            if self.continuous:
                ba_np = np.array([actions[i] for i in idx], dtype=np.float32)
                ba = torch.from_numpy(ba_np).to(bs.device)
                pred = self.policy_net(bs)
                loss = self.policy_criterion(pred, ba)
            else:
                ba_np = np.array([actions[i] for i in idx], dtype=np.float32)
                labels = torch.from_numpy(ba_np.argmax(axis=1).astype(np.int64)).to(
                    bs.device
                )
                logits = self.policy_net(bs)
                loss = self.policy_criterion(logits, labels)
            self.opt_policy.zero_grad()
            loss.backward()
            self.opt_policy.step()

    def update_idm(self, states, next_states, actions):
        self.idm_net.train()
        for idx in util.get_shuffle_idx(len(states), self.batch_size):
            bs_np = np.array([states[i] for i in idx], dtype=np.float32)
            bs = torch.from_numpy(bs_np).to(self.idm_net.net[0].weight.device)
            bns_np = np.array([next_states[i] for i in idx], dtype=np.float32)
            bns = torch.from_numpy(bns_np).to(bs.device)
            if self.continuous:
                ba_np = np.array([actions[i] for i in idx], dtype=np.float32)
                ba = torch.from_numpy(ba_np).to(bs.device)
                pred = self.idm_net(bs, bns)
                loss = self.idm_criterion(pred, ba)
            else:
                ba_np = np.array([actions[i] for i in idx], dtype=np.float32)
                labels = torch.from_numpy(ba_np.argmax(axis=1).astype(np.int64)).to(
                    bs.device
                )
                logits = self.idm_net(bs, bns)
                loss = self.idm_criterion(logits, labels)
            self.opt_idm.zero_grad()
            loss.backward()
            self.opt_idm.step()

    def get_policy_loss(self, states, actions):
        self.policy_net.eval()
        with torch.no_grad():
            s_np = np.array(states, dtype=np.float32)
            s = torch.from_numpy(s_np).to(self.policy_net.net[0].weight.device)
            if self.continuous:
                a_np = np.array(actions, dtype=np.float32)
                a = torch.from_numpy(a_np).to(s.device)
                pred = self.policy_net(s)
                loss = self.policy_criterion(pred, a)
            else:
                a_np = np.array(actions, dtype=np.float32)
                labels = torch.from_numpy(a_np.argmax(axis=1).astype(np.int64)).to(
                    s.device
                )
                logits = self.policy_net(s)
                loss = self.policy_criterion(logits, labels)
            return loss.item()

    def get_idm_loss(self, states, next_states, actions):
        self.idm_net.eval()
        with torch.no_grad():
            s_np = np.array(states, dtype=np.float32)
            s = torch.from_numpy(s_np).to(self.idm_net.net[0].weight.device)
            ns_np = np.array(next_states, dtype=np.float32)
            ns = torch.from_numpy(ns_np).to(s.device)
            if self.continuous:
                a_np = np.array(actions, dtype=np.float32)
                a = torch.from_numpy(a_np).to(s.device)
                pred = self.idm_net(s, ns)
                loss = self.idm_criterion(pred, a)
            else:
                a_np = np.array(actions, dtype=np.float32)
                labels = torch.from_numpy(a_np.argmax(axis=1).astype(np.int64)).to(
                    s.device
                )
                logits = self.idm_net(s, ns)
                loss = self.idm_criterion(logits, labels)
            return loss.item()

    def post_demonstration(self):
        raise NotImplementedError

    def eval_rwd_policy(self, display: bool = False):
        raise NotImplementedError
