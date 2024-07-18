import gym
import numpy as np
import matplotlib.pyplot as plt

from q_learning2 import Q_Learning

env = gym.make('CartPole-v1')
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


#EXPERIMIENTACION
alpha = 0.1
gamma = 0.9
epsilon = 0.2
numberEpisodes = 9000
solve_threshold = 195  # Define the threshold for considering the problem solved

Q1 = Q_Learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds, solve_threshold)
Q1.simulateEpisodes()

plt.figure(figsize=(12, 5))
plt.plot(Q1.sumRewardsEpisode, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.title('Sum of Rewards per Episode')
plt.show()
plt.savefig('convergence.png')

window_size = 100
moving_avg = Q1.moving_average(Q1.sumRewardsEpisode, window_size)

plt.figure(figsize=(12, 5))
plt.plot(moving_avg, color='blue', linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Average Duration (Last 100 Episodes)')
plt.title('Average Duration over 100 Consecutive Episodes')
plt.show()
plt.savefig('average_duration.png')

(obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()
env1.close()

(obtainedRewardsRandom, env2) = Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom, bins=20)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.title('Histogram of Rewards for Random Strategy')
plt.savefig('histogram.png')
plt.show()
env2.close()
