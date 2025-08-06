# rl_environment.py

import gymnasium as gym
import numpy as np
import pandas as pd

class ChurnEnv(gym.Env):
    def __init__(self, user_data: pd.DataFrame, churn_predictor):
        super(ChurnEnv, self).__init__()

        # --- FIX: Use column names from your CSV ---
        self.required_columns = ['age', 'monthly_fee', 'watch_hours']
        
        if not all(col in user_data.columns for col in self.required_columns):
            missing_cols = [col for col in self.required_columns if col not in user_data.columns]
            raise ValueError(f"The user_data is missing required columns: {missing_cols}")

        self.df = user_data[self.required_columns].copy()
        self.churn_predictor = churn_predictor
        self.current_user_index = 0

        self.action_space = gym.spaces.Discrete(4)
        self.action_map = {0: 'promo', 1: 'no_promo', 2: 'email', 3: 'call'}

        low = self.df.min().values.astype(np.float32)
        high = self.df.max().values.astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _get_obs(self):
        return self.df.iloc[self.current_user_index].values.astype(np.float32)

    def _simulate_churn(self, action):
        user_features = self._get_obs()
        model_input = np.copy(user_features)

        # Indices now correspond to: 0: age, 1: monthly_fee, 2: watch_hours
        if self.action_map[action] == 'promo':
            model_input[1] = max(0, model_input[1] * 0.8) # 20% fee reduction
        elif self.action_map[action] == 'call':
             model_input[2] = min(self.observation_space.high[2], model_input[2] * 1.1) # 10% activity increase

        prob_churn = self.churn_predictor.predict_proba(model_input.reshape(1, -1))[0, 1]

        return 1 if np.random.rand() >= prob_churn else -1

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_user_index = np.random.randint(0, len(self.df))
        observation = self._get_obs()
        info = {'user_id': self.df.index[self.current_user_index]}
        return observation, info

    def step(self, action):
        reward = self._simulate_churn(action)
        terminated = True
        truncated = False
        observation = self._get_obs()
        info = {'user_id': self.df.index[self.current_user_index]}
        return observation, reward, terminated, truncated, info