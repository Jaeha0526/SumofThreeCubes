import numpy as np
import torch as t
from torch.nn import functional as F
import os
import time
import sys
from dataclasses import dataclass
from tqdm import tqdm
from numpy.random import Generator
from torch import Tensor
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import einops
from pathlib import Path
from typing import List, Tuple, Literal, Union, Optional
from jaxtyping import Float, Int
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass, field
import math
from collections import Counter
from environments.env1 import SumofCubesEnv

device = t.device("cuda" if t.cuda.is_available() else "cpu")


# This code is mostly from the Arena chapter 2.3 PPO 


@dataclass
class PPOArgs:

    # Basic / global
    seed: int = 1
    cuda: bool = t.cuda.is_available()

    # Duration of different phases
    total_timesteps: int = 500000
    num_envs: int = 4
    num_steps: int = 128
    num_minibatches: int = 4
    batches_per_learning_phase: int = 4

    # k_max
    max_k: int = 100

    # Optimization hyperparameters
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5

    # Computing advantage function
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Computing other loss functions
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.25

    # for env2
    target_k : int = 1
    reward : list = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    max_z : int = 140
    mode : int = 1
    action_bias : list = field(default_factory=lambda: [-1,-2])

    # for env3
    max_n : int = 100
    min_n : int = 1
    ep_length : int = 200

    # for env4
    i_exp : int = 5

    # for env6
    depreciation : int = 100

    # for env8
    plist : list = field(default_factory=lambda: [5, 7, 11])

    def __post_init__(self):
        self.batch_size = self.num_steps * self.num_envs
        assert self.batch_size % self.num_minibatches == 0, "batch_size must be divisible by num_minibatches"
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.total_phases = self.total_timesteps // self.batch_size
        self.total_training_steps = self.total_phases * self.batches_per_learning_phase * self.num_minibatches


# Actor-Critic Network
def layer_init(layer: nn.Linear, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_actor_and_critic_classic(envs: SumofCubesEnv):
    '''
    Returns (actor, critic) in the "classic-control" case, according to diagram above.
    '''
    num_obs = envs.single_observation_space.shape[0]
    num_actions = envs.num_actions

    critic = nn.Sequential(
        layer_init(nn.Linear(num_obs, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 1), std=1.0)
    )

    actor = nn.Sequential(
        layer_init(nn.Linear(num_obs, 128)),
        nn.Tanh(),
        layer_init(nn.Linear(128, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, 256)),
        nn.Tanh(),
        layer_init(nn.Linear(256, num_actions), std=0.01)
    )
    return actor.to(device), critic.to(device)


# memory setting
@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.
    next_value: shape (env,)
    next_done: shape (env,)
    rewards: shape (buffer_size, env)
    values: shape (buffer_size, env)
    dones: shape (buffer_size, env)
    Return: shape (buffer_size, env)
    '''
    # SOLUTION
    T = values.shape[0]
    next_values = t.concat([values[1:], next_value.unsqueeze(0)])
    next_dones = t.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values
    advantages = t.zeros_like(deltas)
    advantages[-1] = deltas[-1]
    for s in reversed(range(1, T)):
        advantages[s-1] = deltas[s-1] + gamma * gae_lambda * (1.0 - dones[s]) * advantages[s]
    return advantages

def minibatch_indexes(rng: Generator, batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''
    Return a list of length num_minibatches = (batch_size // minibatch_size), where each element is an
    array of indexes into the batch. Each index should appear exactly once.

    To relate this to the diagram above: if we flatten the non-shuffled experiences into:

        [1,1,1,1,2,2,2,2,3,3,3,3]

    then the output of this function could be the following list of arrays:

        [array([0,5,4,3]), array([11,6,7,8]), array([1,2,9,10])]

    which would give us the minibatches seen in the first row of the diagram above:

        [array([1,2,2,1]), array([3,2,2,3]), array([1,1,3,3])]
    '''
    assert batch_size % minibatch_size == 0
    # SOLUTION
    indices = rng.permutation(batch_size)
    indices = einops.rearrange(indices, "(mb_num mb_size) -> mb_num mb_size", mb_size=minibatch_size)
    return list(indices)

def to_numpy(arr: Union[np.ndarray, Tensor]):
    '''
    Converts a (possibly cuda and non-detached) tensor to numpy array.
    '''
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


@dataclass
class ReplayMinibatch:
    '''
    Samples from the replay memory, converted to PyTorch for use in neural network training.

    Data is equivalent to (s_t, a_t, logpi(a_t|s_t), A_t, A_t + V(s_t), d_{t+1})
    '''
    observations: Tensor # shape [minibatch_size, *observation_shape]
    actions: Tensor # shape [minibatch_size, ]
    logprobs: Tensor # shape [minibatch_size,]
    advantages: Tensor # shape [minibatch_size,]
    returns: Tensor # shape [minibatch_size,]
    dones: Tensor # shape [minibatch_size,]

class ReplayMemory:
    '''
    Contains buffer; has a method to sample from it to return a ReplayMinibatch object.
    '''
    rng: Generator
    observations: np.ndarray # shape [buffer_size, num_envs, *observation_shape]
    actions: np.ndarray # shape [buffer_size, num_envs]
    logprobs: np.ndarray # shape [buffer_size, num_envs]
    values: np.ndarray # shape [buffer_size, num_envs]
    rewards: np.ndarray # shape [buffer_size, num_envs]
    dones: np.ndarray # shape [buffer_size, num_envs]

    def __init__(self, args: PPOArgs, envs: SumofCubesEnv):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.num_envs = envs.num_envs
        self.obs_shape = envs.single_observation_space.shape
        # self.action_shape = envs.single_action_space.shape
        self.reset_memory()


    def reset_memory(self):
        '''
        Resets all stored experiences, ready for new ones to be added to memory.
        '''
        self.observations = np.empty((0, self.num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, self.num_envs,), dtype=np.int32)
        self.logprobs = np.empty((0, self.num_envs), dtype=np.float32)
        self.values = np.empty((0, self.num_envs), dtype=np.float32)
        self.rewards = np.empty((0, self.num_envs), dtype=np.float32)
        self.dones = np.empty((0, self.num_envs), dtype=bool)


    def add(self, obs, actions, logprobs, values, rewards, dones) -> None:
        '''
        Each argument can be a PyTorch tensor or NumPy array.

        obs: shape (num_environments, *observation_shape)
            Observation before the action
        actions: shape (num_environments, *action_shape)
            Action chosen by the agent
        logprobs: shape (num_environments,)
            Log probability of the action that was taken (according to old policy)
        values: shape (num_environments,)
            Values, estimated by the critic (according to old policy)
        rewards: shape (num_environments,)
            Reward after the action
        dones: shape (num_environments,)
            If True, the episode ended and was reset automatically
        '''
        assert obs.shape == (self.num_envs, *self.obs_shape)
        assert actions.shape == (self.num_envs, )
        assert logprobs.shape == (self.num_envs,)
        assert values.shape == (self.num_envs,)
        assert dones.shape == (self.num_envs,)
        assert rewards.shape == (self.num_envs,)

        self.observations = np.concatenate((self.observations, to_numpy(obs[None, :])))
        self.actions = np.concatenate((self.actions, to_numpy(actions[None, :])))
        self.logprobs = np.concatenate((self.logprobs, to_numpy(logprobs[None, :])))
        self.values = np.concatenate((self.values, to_numpy(values[None, :])))
        self.rewards = np.concatenate((self.rewards, to_numpy(rewards[None, :])))
        self.dones = np.concatenate((self.dones, to_numpy(dones[None, :])))


    def get_minibatches(self, next_value: t.Tensor, next_done: t.Tensor) -> List[ReplayMinibatch]:
        minibatches = []

        # Stack all experiences, and move them to our device
        obs, actions, logprobs, values, rewards, dones = [t.from_numpy(exp).to(device) for exp in [
            self.observations, self.actions, self.logprobs, self.values, self.rewards, self.dones
        ]]

        # Compute advantages and returns (then get the list of tensors, in the right order to add to our ReplayMinibatch)
        advantages = compute_advantages(next_value, next_done, rewards, values, dones.float(), self.args.gamma, self.args.gae_lambda)
        returns = advantages + values
        replay_memory_data = [obs, actions, logprobs, advantages, returns, dones]

        # Generate `batches_per_learning_phase` sets of minibatches (each set of minibatches is a shuffled permutation of
        # all the experiences stored in memory)
        for _ in range(self.args.batches_per_learning_phase):

            indices_for_each_minibatch = minibatch_indexes(self.rng, self.args.batch_size, self.args.minibatch_size)

            for indices_for_minibatch in indices_for_each_minibatch:
                minibatches.append(ReplayMinibatch(*[
                    arg.flatten(0, 1)[indices_for_minibatch] for arg in replay_memory_data
                ]))

        # Reset memory, since we only run this once per learning phase
        self.reset_memory()

        return minibatches


# Construct PPO Agent
class PPOAgent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, args: PPOArgs, envs: SumofCubesEnv):
        super().__init__()
        self.args = args
        self.envs = envs

        # Keep track of global number of steps taken by agent
        self.step = 0

        # Get actor and critic networks
        self.actor, self.critic = get_actor_and_critic_classic(envs)

        # Define our first (obs, done), so we can start adding experiences to our replay memory
        self.next_obs = t.tensor(envs.reset()).to(device, dtype=t.float)
        self.next_done = t.zeros(envs.num_envs).to(device, dtype=t.float)

        # Create our replay memory
        self.memory = ReplayMemory(args, envs)


    def play_step(self) -> List[dict]:
        '''
        Carries out a single interaction step between the agent and the environment, and adds results to the replay memory.

        Returns the list of info dicts returned from `self.envs.step`.
        '''
		    # Get newest observations (i.e. where we're starting from)
        obs = self.next_obs
        dones = self.next_done

        # SOLUTION
        # Compute logits based on newest observation, and use it to get an action distribution we sample from
        with t.inference_mode():
            logits = self.actor(obs)
        probs = Categorical(logits=logits)
        actions = probs.sample()

        # Step environment based on the sampled action
        next_obs, rewards, next_dones, infos = self.envs.step(actions.cpu().numpy())

        # Calculate logprobs and values, and add this all to replay memory
        logprobs = probs.log_prob(actions)
        with t.inference_mode():
            values = self.critic(obs).flatten()

        # change it into x, y, z action form
        self.memory.add(obs, actions, logprobs, values, rewards, dones)

        # Set next observation, and increment global step counter
        self.next_obs = t.from_numpy(next_obs).to(device, dtype=t.float)
        self.next_done = t.from_numpy(next_dones).to(device, dtype=t.float)
        self.step += self.envs.num_envs

        # Return infos dict, for logging
        return obs, infos


    def get_minibatches(self) -> list[ReplayMinibatch]:
        '''
        Gets minibatches from the replay memory.
        '''
        with t.inference_mode():
            next_value = self.critic(self.next_obs).flatten()
        return self.memory.get_minibatches(next_value, self.next_done)


# Objective calculation
def calc_clipped_surrogate_objective(
    probs: Categorical,
    mb_action: Int[Tensor, "minibatch_size"],
    mb_advantages: Float[Tensor, "minibatch_size"],
    mb_logprobs: Float[Tensor, "minibatch_size"],
    clip_coef: float,
    eps: float = 1e-8
) :
    '''Return the clipped surrogate objective, suitable for maximisation with gradient ascent.

    probs:
        a distribution containing the actor's unnormalized logits of shape (minibatch_size, num_actions)
    mb_action:
        what actions actions were taken in the sampled minibatch
    mb_advantages:
        advantages calculated from the sampled minibatch
    mb_logprobs:
        logprobs of the actions taken in the sampled minibatch (according to the old policy)
    clip_coef:
        amount of clipping, denoted by epsilon in Eq 7.
    eps:
        used to add to std dev of mb_advantages when normalizing (to avoid dividing by zero)
    '''
    assert mb_action.shape == mb_advantages.shape == mb_logprobs.shape
    # SOLUTION
    logits_diff = probs.log_prob(mb_action) - mb_logprobs

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + eps)

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()

def calc_value_function_loss(
    values: Float[Tensor, "minibatch_size"],
    mb_returns: Float[Tensor, "minibatch_size"],
    vf_coef: float
) :
    '''Compute the value function portion of the loss function.

    values:
        the value function predictions for the sampled minibatch (using the updated critic network)
    mb_returns:
        the target for our updated critic network (computed as `advantages + values` from the old network)
    vf_coef:
        the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    assert values.shape == mb_returns.shape
    # SOLUTION
    return 0.5 * vf_coef * (values - mb_returns).pow(2).mean()

def calc_entropy_bonus(probs: Categorical, ent_coef: float):
    '''Return the entropy bonus term, suitable for gradient ascent.

    probs:
        the probability distribution for the current policy
    ent_coef:
        the coefficient for the entropy loss, which weights its contribution to the overall objective function. Denoted by c_2 in the paper.
    '''
    # SOLUTION
    return ent_coef * probs.entropy().mean()


# Optimizer and Scheduler
class PPOScheduler:
    def __init__(self, optimizer: Optimizer, initial_lr: float, end_lr: float, total_training_steps: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.total_training_steps = total_training_steps
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after total_training_steps calls to step, the learning rate is end_lr.

        Do this by directly editing the learning rates inside each param group (i.e. `param_group["lr"] = ...`), for each param
        group in `self.optimizer.param_groups`.
        '''
        # SOLUTION
        self.n_step_calls += 1
        frac = self.n_step_calls / self.total_training_steps
        assert frac <= 1
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr + frac * (self.end_lr - self.initial_lr)

def make_optimizer(agent: PPOAgent, total_training_steps: int, initial_lr: float, end_lr: float) -> tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    optimizer = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5, maximize=True)
    scheduler = PPOScheduler(optimizer, initial_lr, end_lr, total_training_steps)
    return (optimizer, scheduler)


# Trainer
class PPOTrainer:

    def __init__(self, args: PPOArgs, environment:SumofCubesEnv, test_mode = False):
        self.args = args
        # self.envs = SumofCubesEnv(args.max_k, args.num_envs)
        self.envs = environment(args)
        self.agent = PPOAgent(self.args, self.envs).to(device)
        self.optimizer, self.scheduler = make_optimizer(self.agent, self.args.total_training_steps, self.args.learning_rate, 0.0)

        # records
        self.last_episode_len_records = []
        self.last_max_sum_records = []
        self.last_reward_records = []
        self.last_reward_list_records = []
        self.last_reward_sum_records = []
        self.current_states = []

        # print for test or not
        self.test_mode = test_mode


    def rollout_phase(self) -> Optional[int]:
        '''
        This function populates the memory with a new set of experiences, using `self.agent.play_step`
        to step through the environment. It also returns the episode length of the most recently terminated
        episode (used in the progress bar readout).
        '''
        # SOLUTION
        last_episode_len = None
        last_max_sum = None
        last_state = None
        last_reward = None
        last_reward_list = None
        last_reward_sum = None
        current_state = []

        for step in range(self.args.num_steps):
            obs, infos = self.agent.play_step()
            for info in infos:
                if "episode" in info.keys():
                    last_episode_len = info["episode"]["l"]
                    # last_episode_return = info["episode"]["r"]
                if "max_sum" in info.keys():
                    last_max_sum = info["max_sum"]
                if "reward" in info.keys():
                    last_reward = info["reward"]
                if "reward_list" in info.keys():
                    last_reward_list = info["reward_list"]
                if "reward_sum" in info.keys():
                    last_reward_sum = info["reward_sum"]
                if "states" in info.keys():
                    current_state.append(info["states"][0])

            if self.test_mode :
                print(infos)
        return last_episode_len, last_max_sum, last_reward, last_reward_list, last_reward_sum, current_state


    def learning_phase(self) -> None:
        '''
        This function does the following:

            - Generates minibatches from memory
            - Calculates the objective function, and takes an optimization step based on it
            - Clips the gradients (see detail #11)
            - Steps the learning rate scheduler
        '''
        # SOLUTION
        minibatches = self.agent.get_minibatches()
        for minibatch in minibatches:
            objective_fn = self.compute_ppo_objective(minibatch)
            objective_fn.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()


    def compute_ppo_objective(self, minibatch: ReplayMinibatch) :
        '''
        Handles learning phase for a single minibatch. Returns objective function to be maximized.
        '''
        # SOLUTION
        logits = self.agent.actor(minibatch.observations)
        probs = Categorical(logits=logits)
        values = self.agent.critic(minibatch.observations).squeeze()

        clipped_surrogate_objective = calc_clipped_surrogate_objective(probs, minibatch.actions, minibatch.advantages, minibatch.logprobs, self.args.clip_coef)
        value_loss = calc_value_function_loss(values, minibatch.returns, self.args.vf_coef)
        entropy_bonus = calc_entropy_bonus(probs, self.args.ent_coef)

        total_objective_function = clipped_surrogate_objective - value_loss + entropy_bonus

        with t.inference_mode():
            newlogprob = probs.log_prob(minibatch.actions)
            logratio = newlogprob - minibatch.logprobs
            ratio = logratio.exp()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs = [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]
        # if self.args.use_wandb: wandb.log(dict(
        #     total_steps = self.agent.step,
        #     values = values.mean().item(),
        #     learning_rate = self.scheduler.optimizer.param_groups[0]["lr"],
        #     value_loss = value_loss.item(),
        #     clipped_surrogate_objective = clipped_surrogate_objective.item(),
        #     entropy = entropy_bonus.item(),
        #     approx_kl = approx_kl,
        #     clipfrac = np.mean(clipfracs)
        # ), step=self.agent.step)

        return total_objective_function


    def train(self) -> None:

        # if args.use_wandb: wandb.init(
        #     project=self.args.wandb_project_name,
        #     entity=self.args.wandb_entity,
        #     name=self.run_name,
        #     monitor_gym=self.args.capture_video
        # )

        progress_bar = tqdm(range(self.args.total_phases))

        for epoch in progress_bar:

            last_episode_len, last_max_sum, last_reward, last_reward_list, last_reward_sum, current_state = self.rollout_phase()
            self.last_episode_len_records.append(last_episode_len)
            self.last_max_sum_records.append(last_max_sum)
            self.last_reward_records.append(last_reward)
            self.last_reward_list_records.append(last_reward_list)
            self.last_reward_sum_records.append(last_reward_sum)
            self.current_states.append(current_state)
            # num_solution = len(self.envs.solutions)

            if last_episode_len is not None:
                progress_bar.set_description(f"Epoch {epoch:02}, Episode length: {last_episode_len}, Max sum: {last_max_sum}, Last reward sum: {last_reward_sum}, current state: {current_state[-1]}")
                # progress_bar.set_description(f"Epoch {epoch:02}, Episode length: {last_episode_len}, Number of Solutions: {num_solution}")
                # progress_bar.set_description(f"Epoch {epoch:02}, Last reward: {last_reward}")

            self.learning_phase()

        # self.envs.close()
        # if self.args.use_wandb:
        #     wandb.finish()
        
        
# plot
def plot_rewards(reward_records, average_only=False):
    # Generate recent 50 interval average
    average_reward = np.empty((len(reward_records),0))
    reward_records = np.array(reward_records)

    for idx in range(len(reward_records[0])):
        # avg_list = np.empty(shape=(len(reward_records),1), dtype=int)
        if idx < 50:
            avg_list = reward_records[:,:idx+1]
        else:
            avg_list = reward_records[:,idx-49:idx+1]
        average_reward = np.hstack([average_reward, np.average(avg_list, axis=-1, keepdims=True)])

    for i in range(len(reward_records)):
        if not average_only :
            plt.plot(reward_records[i])
        plt.plot(average_reward[i,:])

    plt.show()
    return