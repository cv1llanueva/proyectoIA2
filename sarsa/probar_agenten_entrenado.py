import pickle
import gym
import matplotlib.pyplot as plt

# Load the trained SARSA model
with open('trained_sarsa_model.pkl', 'rb') as f:
    Q1 = pickle.load(f)

# Simulate the random strategy
obtainedRewardsRandom, env1 = Q1.simulateLearnedStrategy()



env1.close()
