# train_rl.py

import pandas as pd
import pickle
import numpy as np
from src.rl_environment import ChurnEnv
from src.rl_agent import QTableAgent, DQNAgent, discretize_state

# --- Configuration ---
USE_DQN = True # Set to False to use Q-Table
NUM_EPISODES = 10000
EPISODE_LOG_INTERVAL = 100

# --- Load Data and Pre-trained Model ---
print("Loading data and churn predictor...")
# Make sure these files exist in the specified paths
user_data = pd.read_csv('data/user_data.csv', index_col='user_id')
with open('models/churn_predictor.pkl', 'rb') as f:
    churn_predictor = pickle.load(f)

# --- Initialize Environment and Agent ---
print("Initializing environment and agent...")
env = ChurnEnv(user_data, churn_predictor)

if USE_DQN:
    print("Using DQN Agent.")
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
else:
    print("Using Q-Table Agent.")
    # <<< FIX: Binning now correctly uses the features from the environment: 'age', 'fee', 'activity'
    age_bins = pd.cut(user_data['age'], bins=10, retbins=True)[1]
    fee_bins = pd.cut(user_data['fee'], bins=10, retbins=True)[1]
    activity_bins = pd.cut(user_data['activity'], bins=10, retbins=True)[1]

    state_bins = [age_bins, fee_bins, activity_bins]
    
    # <<< FIX: The QTableAgent in rl_agent.py takes state_bins directly
    agent = QTableAgent(state_bins=state_bins, action_size=env.action_space.n)


# --- Training Loop ---
print(f"Starting training for {NUM_EPISODES} episodes...")
total_rewards = []

for e in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    total_episode_reward = 0

    # An episode consists of a single step in this environment
    while not done:
        # <<< FIX: Call the correct method 'choose_action' from rl_agent.py
        action = agent.choose_action(state)

        # Environment takes a step based on the action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Agent learns from the experience
        if USE_DQN:
            # For DQN, we store the experience in memory
            agent.remember(state, action, reward, next_state, done)
        else:
            # For Q-Table, we learn directly from the step
            # <<< FIX: Call the correct method 'learn' from rl_agent.py
            agent.learn(state, action, reward, next_state)

        state = next_state
        total_episode_reward += reward

    # For DQN, perform experience replay after enough experiences are collected
    if USE_DQN:
        # <<< FIX: Call the correct method 'learn' which handles replay
        agent.learn(batch_size=64)

    total_rewards.append(total_episode_reward)

    # Log progress
    if (e + 1) % EPISODE_LOG_INTERVAL == 0:
        avg_reward = np.mean(total_rewards[-EPISODE_LOG_INTERVAL:])
        epsilon_val = agent.epsilon
        print(f"Episode: {e+1}/{NUM_EPISODES} | "
              f"Avg Reward (Last {EPISODE_LOG_INTERVAL}): {avg_reward:.2f} | "
              f"Epsilon: {epsilon_val:.4f}")

# --- Save Trained Agent ---
print("\nTraining finished.")
print("Saving trained agent...")

output_path = f"models/{'dqn_agent.pth' if USE_DQN else 'q_table_agent.npy'}"
# <<< FIX: Call the correct method 'save' from rl_agent.py
agent.save(output_path)

print(f"Agent saved successfully to {output_path}")
env.close()