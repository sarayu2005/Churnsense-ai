# src/rl_agent.py

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

### --- Basic: Q-Table Agent --- ###

def discretize_state(state, bins):
    """Converts a continuous state into a discrete one for the Q-table."""
    discretized = []
    for i, feature in enumerate(state):
        discretized.append(np.digitize(feature, bins[i]) - 1)
    return tuple(discretized)

class QTableAgent:
    def __init__(self, state_bins, action_size, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_bins = state_bins
        # Calculate table dimensions from bins: e.g., [10 age bins, 10 fee bins, 5 activity bins]
        table_dims = [len(b) + 1 for b in state_bins] + [action_size]
        self.q_table = np.zeros(table_dims)
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        discrete_s = discretize_state(state, self.state_bins)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size) # Explore
        return np.argmax(self.q_table[discrete_s]) # Exploit

    def learn(self, state, action, reward, next_state):
        discrete_s = discretize_state(state, self.state_bins)
        discrete_next_s = discretize_state(next_state, self.state_bins)
        
        old_value = self.q_table[discrete_s][action]
        next_max = np.max(self.q_table[discrete_next_s])
        
        # Q-learning formula: Q(s,a) = Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[discrete_s][action] = new_value

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)

### --- Advanced: DQN Agent --- ###

class QNetwork(nn.Module):
    """Simple Feed-Forward Neural Network for Q-value approximation."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000) # Replay buffer

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size) # Explore
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        return np.argmax(action_values.cpu().data.numpy()) # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return # Not enough experiences to learn from
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Get Q-values for current states from the Q-network
        q_predicted = self.q_network(states).gather(1, actions)

        # Get max Q-values for next states
        q_next_max = self.q_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute target Q-value: R + gamma * max_a' Q(s',a')
        # If the episode is done, the target is just the reward
        q_target = rewards + (self.gamma * q_next_max * (1 - dones))

        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(q_predicted, q_target)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath))
        self.q_network.eval() # Set to evaluation mode