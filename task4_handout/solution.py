import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode

class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    Simple MLP Q-function.
    '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)
        self.Q1_net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])
        self.Q2_net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])
        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # TODO: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)
        
        # value = torch.zeros(x.shape[0], 1, device=x.device)

        #####################################################################
        input = torch.cat((x, a), dim=1)
        q1 = self.Q1_net(input)
        q2 = self.Q2_net(input)
        return q1, q2


    def Q1(self, x, a):
        input = torch.cat((x, a), dim=1)
        q1 = self.Q1_net(input)
        return q1




class Actor(nn.Module):
    '''
    Simple Tanh deterministic actor.
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: add components as needed (if needed)
        self.lr = 1
        self.net = MLP([obs_size] + ([num_units] * num_layers) + [action_size])
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        self.action_high = action_high.to(Agent.device)
        self.action_low = action_low.to(Agent.device)
        #####################################################################
        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # TODO: code the forward pass
        # the actor will receive a batch of observations of shape (batch_size x obs_size)
        # and output a batch of actions of shape (batch_size x action_size)

        # action = torch.zeros(x.shape[0], self.action_scale.shape[-1], device=x.device)

        #####################################################################
        return self.action_high * torch.tanh(self.net(x))


class Agent:

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor, 
    exploration_noise: float = 0.1  # epsilon for epsilon-greedy exploration
    c = 0.1
    d = 2
    tau = 0.005
    
    #########################################################################

    def __init__(
            self,
            env,
    ):
        print(env)

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float().to(Agent.device)
        self.action_high = torch.tensor(env.action_space.high).float().to(Agent.device)

        #####################################################################
        # TODO: initialize actor, critic and attributes
        self.actor = Actor(
            action_low=self.action_low,
            action_high=self.action_high,
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=3,
            num_units=128
        ).to(device=Agent.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=3,
            num_units=128
        ).to(device=Agent.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.t = 1



        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)


        #####################################################################
        with torch.no_grad():
            epsilon = (torch.randn_like(action) * self.exploration_noise).clamp(-self.c, self.c)
            a_tilde = (self.actor_target(next_obs) + epsilon).clamp(self.action_low, self.action_high)

            target_Q1, target_Q2 = self.critic_target(next_obs, a_tilde)
            target_Q = torch.minimum(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(dim=-1) + (1 - done).unsqueeze(dim=-1) * self.gamma * target_Q

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.t % self.d == 0:
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.t += 1

        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # TODO: return the agent's action for an observation (np.array
        # of shape (obs_size, )). The action should be a np.array of
        # shape (act_size, )

        action = self.actor(torch.Tensor(obs).to(device=Agent.device)).cpu().numpy()

        #####################################################################
        return action

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
