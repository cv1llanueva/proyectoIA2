# Note: 
# You can either use gym (not maintained anymore) or gymnasium (maintained version of gym)    
     
# tested on     
# gym==0.26.2
# gym-notices==0.0.8
 
#gymnasium==0.27.0
#gymnasium-notices==0.0.1
 
# classical gym 
import gym
# instead of gym, import gymnasium 
# import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt 
 
# import the class that implements the Q-Learning algorithm
from q_learning2 import Q_Learning
 
#env=gym.make('CartPole-v1',render_mode='human')
env=gym.make('CartPole-v1')
(state,_)=env.reset()
#env.render()
#env.close()
 
# here define the parameters for state discretization
upperBounds=env.observation_space.high
lowerBounds=env.observation_space.low
cartVelocityMin=-3
cartVelocityMax=3
poleAngleVelocityMin=-10
poleAngleVelocityMax=10
upperBounds[1]=cartVelocityMax
upperBounds[3]=poleAngleVelocityMax
lowerBounds[1]=cartVelocityMin
lowerBounds[3]=poleAngleVelocityMin
 
numberOfBinsPosition=30
numberOfBinsVelocity=30
numberOfBinsAngle=30
numberOfBinsAngleVelocity=30
numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]
 
# define the parameters
alpha=0.1
gamma=0.999
epsilon=0.1
numberEpisodes=10000
 
# create an object
Q1=Q_Learning(env,alpha,gamma,epsilon,numberEpisodes,numberOfBins,lowerBounds,upperBounds)
# run the Q-Learning algorithm
Q1.simulateEpisodes()
# simulate the learned strategy
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()
 
plt.figure(figsize=(12, 5))
# plot the figure and adjust the plot parameters
plt.plot(Q1.sumRewardsEpisode,color='blue',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.yscale('log')
plt.show()
plt.savefig('convergence.png')
 
 
# close the environment
env1.close()
# get the sum of rewards
np.sum(obtainedRewardsOptimal)
 
# now simulate a random strategy
(obtainedRewardsRandom,env2)=Q1.simulateRandomStrategy()
plt.hist(obtainedRewardsRandom)
plt.xlabel('Sum of rewards')
plt.ylabel('Percentage')
plt.savefig('histogram.png')
plt.show()
 
# run this several times and compare with a random learning strategy
(obtainedRewardsOptimal,env1)=Q1.simulateLearnedStrategy()
