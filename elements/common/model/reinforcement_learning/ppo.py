import torch


class PPO:
    def __init__(self, policy, policy_old, dev="cuda:0"):

        self.dev = dev

        self.policy = policy
        self.policy_old = policy_old

        self.policy_old.load_state_dict(self.policy.state_dict())

    @torch.no_grad()
    def select_action(self, state):

        state = torch.FloatTensor(state).to(self.dev)
        action, action_logprob = self.policy_old.act(state)

        return action, action_logprob

    def evaluate(self, state, action):
        return self.policy.evaluate(state, action)

    def replace_old_policy(self):
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

