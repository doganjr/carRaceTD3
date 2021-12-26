import numpy as np
import torch

class Memory():
    def __init__(self, mem_size, batch_size, state_dim, action_dim, device, exploration_steps):

        self.exploration_steps = exploration_steps
        self.device = device
        self.mem_size = mem_size + self.exploration_steps
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.state_memory = torch.zeros((self.mem_size, state_dim)).to(self.device)
        self.next_state_memory = torch.zeros((self.mem_size, state_dim)).to(self.device)
        self.action_memory = torch.zeros((self.mem_size, action_dim)).to(self.device)
        self.reward_memory = torch.zeros(self.mem_size).to(self.device)
        self.terminal_memory = torch.zeros(self.mem_size).to(self.device)
        self.mem_index = 0
        self.mem_full = 0

        self.gamma = 0.99

    def store(self, state, next_state, action, reward, terminal):
        index = self.mem_index % self.mem_size
        self.state_memory[index] = torch.tensor(state, device=self.device)
        self.next_state_memory[index] = torch.tensor(next_state, device=self.device)
        self.action_memory[index] = torch.tensor(action, device=self.device)
        self.reward_memory[index] = torch.tensor(reward, device=self.device)
        self.terminal_memory[index] = torch.tensor(int(terminal), device=self.device)
        self.mem_index += 1
        self.mem_full += 1
        if self.mem_full >= self.mem_size:
            self.mem_full = self.mem_size
        if self.mem_index == self.mem_size:
            self.mem_index = 0

    def sample(self):
        batch_index = np.random.choice(self.mem_full, self.batch_size, replace=True)
        state_batch = self.state_memory[batch_index]
        next_state_batch = self.next_state_memory[batch_index]
        action_batch = self.action_memory[batch_index]
        reward_batch = self.reward_memory[batch_index]
        terminal_batch = self.terminal_memory[batch_index]

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
