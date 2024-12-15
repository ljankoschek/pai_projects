import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


def interpolate_params(target_net, net, tau):
    with torch.no_grad():  # Disable gradient tracking
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


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
        self.lr = 1
        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

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
        return self.net(input)


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
        self.action_high = action_high
        self.action_low = action_low
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
    # device = 'cuda' if torch.cuda.is_available() else
    device = 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor, 
    exploration_noise: float = 0.1  # epsilon for epsilon-greedy exploration
    
    #########################################################################

    def __init__(self, env):
        print(env)

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()

        #####################################################################
        # TODO: initialize actor, critic and attributes
        self.actor = Actor(
            action_low=self.action_low,
            action_high=self.action_high,
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=10,
            num_units=64
        )
        self.critic1 = Critic(
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=10,
            num_units=64
        )

        self.critic2 = Critic(
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=10,
            num_units=64
        )

        self.target1 = Critic(
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=10,
            num_units=64
        )

        self.target2 = Critic(
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=10,
            num_units=64
        )

        self.actor_target = Actor(
            action_low=self.action_low,
            action_high=self.action_high,
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=10,
            num_units=64
        )

        self.d = 10
        self.t = 1
        self.tau = 0.1
        self.sigma = 0.5
        self.target_sigma = 0.2
        self.c = 0.5

        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

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
            a_tilde = self.actor_target(next_obs)
            a_tilde += torch.clamp(self.target_sigma * torch.randn(a_tilde.shape), -self.c, self.c)
            a_tilde = torch.clamp(a_tilde, -1, 1)
            target_pred1 = self.target1(next_obs, a_tilde)
            target_pred2 = self.target2(next_obs, a_tilde)
            done = done.unsqueeze(dim=-1)  # Ensure done has the correct shape
            y = reward.unsqueeze(dim=-1) + (1 - done) * self.gamma * torch.minimum(target_pred1, target_pred2)
        pred1 = self.critic1(obs, action)
        pred2 = self.critic2(obs, action)
        loss1 = self.critic1.loss(pred1, y.detach())
        loss2 = self.critic2.loss(pred2, y.detach())
        self.critic1.optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        self.critic2.optimizer.zero_grad()
        loss2.backward()

        if self.t % self.d == 0:

            # TODO
            a = self.actor(obs)
            loss = -self.critic1(obs, a).mean()
            self.actor.optimizer.zero_grad()
            loss.backward()

            interpolate_params(self.target1, self.critic1, self.tau)
            interpolate_params(self.target2, self.critic2, self.tau)
            interpolate_params(self.actor_target, self.actor, self.tau)

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

        action = self.actor(torch.Tensor(obs))
        action += self.sigma * torch.randn(action.shape)
        action = torch.clamp(action,-1,1)
        #####################################################################
        return action.numpy()

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
