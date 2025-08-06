# rl_utils.py

import pandas as pd
import pickle
import numpy as np
import os
import torch
import time

from rl_environment import ChurnEnv
from rl_agent import QTableAgent, DQNAgent

USE_DQN = True 
NUM_EPISODES = 10000
EPISODE_LOG_INTERVAL = 100

def train_rl_agent():
    env = None
    try:
        yield "event: message\ndata: Starting Reinforcement Learning training...\n\n"
        
        # --- FIX: Check for the required ML model BEFORE trying to load it ---
        churn_model_path = '../models/churn_predictor.pkl'
        if not os.path.exists(churn_model_path):
            error_msg = "ERROR: The ML model 'churn_predictor.pkl' was not found. Please run the 'ML Prediction' analysis first to generate the model."
            yield f"event: message\ndata: \n\n--- SCRIPT FAILED ---\n{error_msg}\n\n"
            return # Stop the function execution

        yield "event: message\ndata: -> Reading user data file...\n\n"
        user_data = pd.read_csv('../uploads/user_data.csv', index_col='customer_id')
        yield "event: message\ndata: -> User data loaded successfully.\n\n"
        
        yield "event: message\ndata: -> Loading churn_predictor.pkl...\n\n"
        with open(churn_model_path, 'rb') as f:
            churn_predictor = pickle.load(f)
        yield "event: message\ndata: -> churn_predictor.pkl loaded successfully.\n\n"

        yield "event: message\ndata: Initializing RL environment...\n\n"
        env = ChurnEnv(user_data, churn_predictor)
        yield "event: message\ndata: -> Environment initialized.\n\n"
        
        agent_type = "DQN" if USE_DQN else "Q-Table"
        yield f"event: message\ndata: Initializing {agent_type} Agent...\n\n"

        if USE_DQN:
            agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
        else:
            age_bins = pd.cut(user_data['age'], bins=10, retbins=True)[1]
            fee_bins = pd.cut(user_data['monthly_fee'], bins=10, retbins=True)[1]
            activity_bins = pd.cut(user_data['watch_hours'], bins=10, retbins=True)[1]
            state_bins = [age_bins, fee_bins, activity_bins]
            agent = QTableAgent(state_bins=state_bins, action_size=env.action_space.n)
        
        yield "event: message\ndata: -> Agent initialized.\n\n"

        yield f"event: message\ndata: Starting training loop for {NUM_EPISODES} episodes...\n\n"
        total_rewards = []
        for e in range(NUM_EPISODES):
            state, _ = env.reset()
            done = False
            reward = 0
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if USE_DQN:
                    agent.remember(state, action, reward, next_state, done)
                else:
                    agent.learn(state, action, reward, next_state)
                state = next_state
            if USE_DQN and len(agent.memory) > agent.batch_size:
                agent.learn(batch_size=64)
            total_rewards.append(reward)
            if (e + 1) % EPISODE_LOG_INTERVAL == 0:
                avg_reward = np.mean(total_rewards[-EPISODE_LOG_INTERVAL:])
                epsilon_val = agent.epsilon
                log_msg = (f"Episode: {e+1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon_val:.4f}")
                yield f"event: message\ndata: {log_msg}\n\n"
                time.sleep(0.01)
        
        yield "event: message\ndata: \nTraining finished.\n\n"
        yield "event: message\ndata: Saving trained agent...\n\n"
        
        output_path = f"../models/{'dqn_agent.pth' if USE_DQN else 'q_table_agent.npy'}"
        agent.save(output_path)
        yield f"event: message\ndata: Agent saved successfully to {output_path}\n\n"

    except Exception as e:
        yield f"event: message\ndata: \n\n--- SCRIPT FAILED ---\nERROR: {str(e)}\n\n"
    finally:
        if env is not None:
            env.close()
        yield "event: close\ndata: \n\n"


def get_rl_recommendation(user_state: dict):
    try:
        action_map = {0: 'Offer Promo', 1: 'No Action', 2: 'Send Email', 3: 'Call Customer'}
        state_features = ['age', 'monthly_fee', 'watch_hours'] 
        state_array = np.array([user_state[feature] for feature in state_features], dtype=np.float32)

        if USE_DQN:
            model_path = '../models/dqn_agent.pth'
            if not os.path.exists(model_path):
                return "DQN model not found. Please train the agent first."
            
            agent = DQNAgent(state_size=len(state_features), action_size=len(action_map))
            agent.load(model_path)
            agent.epsilon = 0.0 
        else:
            return "Q-Table recommendation is not implemented."

        action_index = agent.choose_action(state_array)
        return action_map.get(action_index, "Unknown Action")

    except Exception as e:
        return f"An error occurred: {str(e)}"
