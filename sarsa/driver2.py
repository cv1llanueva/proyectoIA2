import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
import pickle

# Import the SARSA class
from sarsa import SARSA

env = gym.make('CartPole-v0')
(state, _) = env.reset()

upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

alpha = 0.1
gamma = 0.99
epsilon = 0.1
numberEpisodes = 8000

# Create an object of SARSA
Q1 = SARSA(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)

# Measure execution time and memory usage
start_time = time.time()
tracemalloc.start()

Q1.simulateEpisodes()

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
end_time = time.time()

execution_time = end_time - start_time
max_memory_used = peak / 10**6  # Convert to MB

# Simulate the learned strategy
obtainedRewardsOptimal, env1 = Q1.simulateLearnedStrategy()

# Calculate the mean duration over 100 consecutive episodes
mean_rewards = [np.mean(Q1.sumRewardsEpisode[i:i+100]) for i in range(0, len(Q1.sumRewardsEpisode), 100)]

# Calculate the number of episodes needed to reach an average reward of 195
episodes_to_195 = next((i for i, reward in enumerate(mean_rewards) if reward >= 195), None)

# Calculate the maximum average reward
max_avg_reward = max(mean_rewards)

# Plot individual episode rewards
plt.figure(figsize=(12, 5))
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.title('Individual Episode Rewards')
plt.savefig('individual_rewards.png')
plt.show()

# Plot mean rewards over 100 episodes
plt.figure(figsize=(12, 5))
plt.plot(mean_rewards, color='red', linewidth=1)
plt.xlabel('Episodes (in hundreds)')
plt.ylabel('Mean Reward (over 100 episodes)')
plt.yscale('log')
plt.title('Mean Rewards Over 100 Episodes')
plt.savefig('mean_duration_100_episodes.png')
plt.show()

env1.close()
np.sum(obtainedRewardsOptimal)


obtainedRewardsRandom, env2 = Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()

obtainedRewardsOptimal, env1 = Q1.simulateLearnedStrategy()

with open('trained_sarsa_model.pkl', 'wb') as f:
    pickle.dump(Q1, f)

print(f"Execution time: {execution_time} seconds")
print(f"Max memory used: {max_memory_used} MB")
print(f"Number of episodes to reach an average reward of 495: {episodes_to_195*100 if episodes_to_195 is not None else 'Not reached'}")
