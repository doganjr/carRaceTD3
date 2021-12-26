from network import CriticNetwork, ActorNetwork
from memory import Memory
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import os

class Agent:
    def __init__(self, gamma, tau, actorlr, criticlr, variance, action_dim, mem_size, batch_size, state_dim, env_name, exploration_steps):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exploration_steps = exploration_steps
        self.memory = Memory(mem_size=mem_size, batch_size=batch_size, state_dim=state_dim, action_dim=action_dim, device=self.device, exploration_steps=self.exploration_steps)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.max_action = 1
        self.discount = 0.99

        self.actor = ActorNetwork(state_dim, 300, 300, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=actorlr, weight_decay=2e-6)
        self.actor_target = ActorNetwork(state_dim, 300, 300, action_dim).to(self.device)

        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=criticlr)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_criterion = nn.MSELoss()

        self.variance = variance
        self.env_name = env_name
        self.total_it = 0

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def get_noise(self):
        noise = np.random.normal(0, self.variance, self.action_dim)
        noise = torch.tensor(noise, dtype=torch.float, device=self.device).unsqueeze(0)
        return noise

    def action_selection(self, obs, train_flag):
        pert = self.get_noise()
        action = self.actor.forward(torch.tensor(obs, device=self.device).unsqueeze(0).float()) + float(train_flag) * pert
        return action.squeeze(0).cpu().detach().numpy()

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = self.memory.sample()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(-1) + (1 - done).unsqueeze(-1) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, timestep, id):
        a = self.env_name
        if os.path.isdir("model_comp_ddpg_"+a) == False:
            os.mkdir("model_comp_ddpg_"+a)
        if os.path.isdir("model_comp_ddpg_"+a+"/"+str(id)) == False:
            os.mkdir("model_comp_ddpg_"+a+"/"+str(id))

        torch.save(self.actor.state_dict(), "model_comp_ddpg_"+a+"/"+str(id)+"/"+str(timestep)+"_"+str(id)+"_actor.pth")
        #torch.save(self.critic.state_dict(), "model_comp_ddpg_"+a+"/"+str(id)+"/"+str(timestep)+"_"+str(id)+"_critic.pth")

    def load_models(self, timestep, id):
        a = self.env_name
        self.actor.load_state_dict(torch.load( "model_comp_ddpg_"+a+"/"+str(timestep)+"_"+str(id)+"_actor.pth",  map_location=self.device))
        self.critic.load_state_dict(torch.load("model_comp_ddpg_"+a+"/"+str(timestep)+"_"+str(id)+"_critic.pth",  map_location=self.device))
