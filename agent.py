'''
DDPG Agent Class
'''

import numpy as np
import random
from collections import deque
import os

from model import Actor, Critic
from utils import OUNoise, ReplayBuffer

import torch
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed=0, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, gamma=GAMMA, checkpoint_path='./checkpoints/', pretrained=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.checkpoint_path = checkpoint_path

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        # If pretrained, load weights
        if pretrained:
            actor_dict = torch.load(os.path.join(self.checkpoint_path,'checkpoint_actor.pth'))
            critic_dict = torch.load(os.path.join(self.checkpoint_path,'checkpoint_critic.pth'))
            self.actor_local.load_state_dict(actor_dict)
            self.actor_target.load_state_dict(actor_dict)
            self.critic_local.load_state_dict(critic_dict)
            self.critic_target.load_state_dict(critic_dict)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
    
    def step(self, state, action, reward, next_state, done, tstep=LEARN_EVERY+1):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and tstep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences)
            
    def train(self, env, n_episodes=1000):
        """Deep Deterministic Policy Gradient (DDPG) Learning.

        Params
        ======
            env (UnityEnvironment): Unity environment
            n_episodes (int): maximum number of training episodes
        """
        # create checkpoints folder if necessary
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)
        # get the default brain
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        # last 100 scores
        scores_deque = deque(maxlen=100)
        # list containing scores from each episode
        all_scores = []
        # list containing window averaged scores
        avg_scores = []
        # for each episode
        for i_episode in range(1, n_episodes+1):
            # reset environment
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            # reset noise
            self.reset()
            scores = np.zeros(num_agents) 
            # for each timepoint
            t=0
            while True:
                # agent action
                actions = self.act(states)
                # get the next state
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                # get the reward
                rewards = env_info.rewards
                # see if episode has ended
                dones = env_info.local_done
                # step
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    self.step(state, action, reward, next_state, done, t)
                states = next_states
                scores += rewards
                t+=1
                if np.any(dones):
                    break 
            # save most recent score
            max_score = np.max(scores)
            scores_deque.append(max_score)
            all_scores.append(max_score)
            avg_scores.append(np.mean(scores_deque))
            print('\rEpisode {}\tScore: {:.2f}\tMax Score: {:.2f}'.format(i_episode, max_score, np.mean(scores_deque)), end="")
            if i_episode % 50 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=0.5:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                torch.save(self.actor_local.state_dict(), self.checkpoint_path+'checkpoint_actor.pth')
                torch.save(self.critic_local.state_dict(), self.checkpoint_path+'checkpoint_critic.pth')
                break
            
        return all_scores, avg_scores

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

        self.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def play(self, env, n_episodes=5):
        """Play a few episodes with trained agents.

        Params
        ======
            env (UnityEnvironment): Unity environment
            n_episodes (int): maximum number of training episodes
        """
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        # reset the environment
        env_info = env.reset(train_mode=False)[brain_name]
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        state_size = env_info.vector_observations.shape[1]

        # for each episode
        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations
            self.reset() # set the noise to zero
            score = np.zeros(num_agents)
            while(True):
                actions = self.act(states, add_noise=False)
                env_info = env.step(actions)[brain_name]
                # get the next states
                next_states = env_info.vector_observations             
                # get the rewards
                rewards = env_info.rewards                             
                # see if the episode has finished for any agent
                dones = env_info.local_done                            

                self.step(states, actions, rewards, next_states, dones)
                states = next_states
                score += rewards
                if np.any(dones):
                    break

            print('Best Score:', np.max(score))    
        env.close()
