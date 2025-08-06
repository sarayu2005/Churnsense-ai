🧠 ChurnSense AI
“Upload. Understand. Act.” – An interactive platform for understanding and acting on customer churn data using a full suite of analytical tools, from exploratory analysis to AI-driven action recommendations.

This project is a full-stack web application that allows users to upload a customer dataset and receive a comprehensive analysis, including machine learning predictions, causal inference, survival analysis, and reinforcement learning recommendations to reduce churn.


✨ Features Implemented

This platform provides a complete, end-to-end workflow for churn analysis:

1. 📤 Upload Dataset

Upload customer data in .csv format.

The backend automatically detects key columns like the churn indicator and time-based features for survival analysis.

2. 🧪 Exploratory Data Analysis (EDA)

Get a statistical summary of your dataset.

Visualize missing values.

Generate interactive plots for feature distributions and a correlation heatmap.

Download a comprehensive PDF report of the EDA results.

3. 🤖 Churn Prediction (Machine Learning)

Automatically trains, compares, and selects the best model from a suite of classifiers:

Logistic Regression

Random Forest

XGBoost

Displays key performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

Visualizes model comparisons and ROC curves.

The best-trained model is saved and used by the Reinforcement Learning agent.

4. ⏳ Survival Analysis

Understand when churn is likely to happen.

Kaplan-Meier Curve: Visualizes the probability of customers remaining subscribed over time.

Cox Proportional Hazards: Identifies the features that have the most significant impact on the timing of churn.

Provides a list of high-risk customers based on their survival scores.

5. 🔎 Causal Inference

Go beyond correlation to understand causation.

Uses the DoWhy library to estimate the causal effect of a specific feature (e.g., monthly_fee) on churn.

Generates a causal graph to visualize the relationships between features.

6. 🎮 Action Recommendation (Reinforcement Learning)

The most advanced feature of the platform, designed to answer: "What should I do?"

Train an Agent: Uses a Deep Q-Network (DQN) to learn an optimal policy for churn reduction. The agent is trained in the browser with live log updates.

Get Recommendations: Input a customer's profile (age, fee, activity) to get a specific, AI-driven recommendation from the trained agent (e.g., "Offer Promo," "Send Email").

🛠️ Tech Stack

Frontend: React.js

Backend: FastAPI (Python)

Machine Learning: Scikit-learn, XGBoost

Reinforcement Learning: PyTorch

Causal Inference: DoWhy

Survival Analysis: Lifelines

Data Handling: Pandas

API Server: Uvicorn

📂 Project Structure

churnsense-ai/
├── backend/
│   ├── main.py             # FastAPI application
│   ├── eda_utils.py        # All backend utility scripts...
│   ├── ml_utils.py
│   ├── survival_utils.py
│   ├── causal_utils.py
│   ├── rl_utils.py
│   ├── rl_agent.py
│   └── rl_environment.py
│
├── frontend/
│   ├── src/
│   │   └── App.js          # Main React component
│   └── ...
│
├── uploads/                # Where user-uploaded CSVs are stored
├── models/                 # Where trained ML/RL models are saved
└── plots/                  # Where generated plots are saved

🚀 Getting Started

Follow these instructions to get the project running on your local machine.

Prerequisites:

Python 3.9+

Node.js and npm

1. Clone the Repository

git clone [https://github.com/sarayu2005/churnsense-ai.git](https://github.com/sarayu2005/churnsense-ai.git)
cd churnsense-ai

2. Backend Setup

Navigate to the backend directory and install the required Python packages.

cd backend
pip install -r requirements.txt  # You may need to create this file

(Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt in your activated virtual environment.)

Run the backend server:

uvicorn main:app --reload

The backend will be running at http://127.0.0.1:8000.

3. Frontend Setup

Open a new terminal, navigate to the frontend directory, and install the npm packages.

cd frontend
npm install

Run the frontend development server:

npm start

The React application will open in your browser at http://localhost:3000.

📋 How to Use

Upload Data: Open the web app, click "Upload CSV Dataset," and select your churn data file.

Run Analyses: Use the buttons to run EDA, Survival Analysis, ML Prediction, and Causal Inference. The results will appear in a "tabbed" view.

Train the RL Agent: After running "ML Prediction" (which creates the necessary churn_predictor.pkl model), click "Train RL Agent." Wait for the training process to complete in the log window.

Get Recommendations: Once the RL agent is trained, input a customer's details into the form and click "Get Action" to receive an AI-powered recommendation.

