
from dqn_agent import Agent
from model import QNetwork
from unityagents import UnityEnvironment

import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque

# set the path to match the location of the Unity environment
path = "/home/alain/Documentos/GitHub/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64"
env = UnityEnvironment(file_name=path)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents in the environment
print('Number of agents:', len(env_info.agents))
# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)
print('States have length:', state_size)


agent = Agent(state_size=state_size,action_size=action_size,seed = 0)
n_episodes = 5000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

scores = []
scores_window = deque(maxlen=100)
eps = eps_start
print_every = 10
stop_criteria = 13
# ------------------- begin training ------------------- #
for e in range(1,n_episodes+1):
    # --- New Episode --- #
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # get the current state
    state = env_info.vector_observations[0]
    score = 0
    # --- Visits --- #
    while True:
        # Agent selects an action
        action = agent.choose_action(state,eps)
        # Take the action and get {s',reward,done} from environment
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        # Take step with the agent
        agent.step(state,action,reward,next_state,done)
        # Update monitorization variables & params for next visit
        state = next_state
        score += reward
        if done:
            break
    # Update monitorization variables & params for next Episode
    eps = max(eps_end,eps*eps_decay)
    scores.append(score)
    scores_window.append(score)
    if e % print_every == 0:
        print('Episode {}/{}\tAvg Score: {:.2f}'.format(e,n_episodes,np.mean(scores_window)))
    if np.mean(scores_window) > stop_criteria:
        print('Environment solved in {} episodes'.format(e))
        torch.save(agent.qnetwork_local.state_dict(), 'model_weights_targetnetwork_er.pth')
        break

env.close()

# =============================================================================
# PLOT THE RESULTS
# =============================================================================
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot([0,len(scores)-1],[stop_criteria,stop_criteria],color='#00FF00',label='target')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# plot the scores every 100 episodes
compact_scores = []
for i in range(len(scores)):
    if i <= 100:
        compact_scores.append(np.mean(scores[0:i]))
    else:
        compact_scores.append(np.mean(scores[i-100:i]))
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(compact_scores)), compact_scores, label='agent')
plt.plot([0,len(compact_scores)-1],[stop_criteria,stop_criteria],color='#00FF00',label='target')
plt.title('Score obtained in 100 consecutive episodes')
plt.ylabel('Score')
plt.xlabel('Episodes #')
plt.show()
