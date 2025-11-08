import numpy as np
import torch
from typing import Tuple, Any, List
from .environment import Environment
from .ppo import PPO
from .rolloutbuffer import RolloutBuffer


def reset_environment(environment: Environment) -> np.ndarray:
    """
        Resets the environment

        :param environment: The environment

        :return: The initial state for the next epsiode (np.ndarray)
    """
    return environment.reset()


def create_replay_buffer() -> RolloutBuffer:
    """
        :return: Returns a rolloutbuffer. It stores states, actions, rewards, done and action_log_probabilities
    """
    return RolloutBuffer()


def update_buffer(buffer: RolloutBuffer, state, action, action_logprob, reward, done):
    """
        Appends an experience (state, action, reward, done and action_log_probabilities)

        :param buffer: The rolloutbuffer
        :param state: The state
        :param action: The action
        :param action_logprob: The action_logprob
        :param reward: The reward
        :param done: whether the episode has ended
    """

    buffer.states.append(state)
    buffer.actions.append(action)
    buffer.logprobs.append(action_logprob)
    buffer.rewards.append(reward)
    buffer.is_terminals.append(done)


def buffer_to_tensors(lists: List[Any]) -> List[torch.Tensor]:
    """
        Converts lists to tensors. Needed to prepare the rolloutbuffer for the evaluation step

        :param lists: A list holding the lists

        :return the lists in Tensor format
    """
    return [torch.squeeze(torch.stack(l, dim=0)).detach() for l in lists]


def clear_buffer(buffer: RolloutBuffer):
    """
        Clears the entire rolloutbuffer

        :param buffer: the rolloutbuffer
    """
    buffer.clear()


def compute_monte_carlo_return(rewards: List, is_terminals: List, gamma: float, normalize: bool = True):
    """
        Computes the sum of discounted rewards (return) which is needed to update the policy

        :param rewards: A list holding the rewards
        :param is_terminals: A list holding whether the experience caused the episode to end
        :param gamma: the discount factor
        :param normalize: whether the returns should be normalized

        :return the return
    """
    returns = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        returns.insert(0, discounted_reward)

    # Normalizing the rewards
    if normalize:
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    return returns



def replace_old_policy(ppo_agent: PPO):
    """
        Replaces the old policy of the agent by the new one

        :param ppo_agent: A PPO RL agent
    """
    ppo_agent.replace_old_policy()
