import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def plotResults(scores,checkpoint=0,baseline=13,compacted = True):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if compacted:
        ax.set_ylim([0,17])
        plt.axvline(x=checkpoint,ymax = 13/17,linewidth = 1, color = 'red', linestyle = ':')
        plt.title('Avg score obtained in 100 consecutive episodes')
    else:
        plt.title('Scores obtained in each episode')
    plt.plot(np.arange(len(scores)), scores)
    plt.axhline(y=13, linewidth = 1, color = 'black', linestyle = '--')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def scoresEvery100episodes(scores):
    compact_scores = []
    for i in range(1,len(scores)+1):
        if i < 100:
            compact_scores.append(np.mean(scores[0:i]))
        else:
            compact_scores.append(np.mean(scores[i-100:i]))
    return compact_scores

def analyzeSingleRun(filename):
    with open(filename,'rb') as f:
       scores, checkpoint = pickle.load(f)
    plotResults(scores,checkpoint,compacted = False)
    # plotResults(scoresEvery100episodes(scores),checkpoint)
    print('Problem solved in {} episodes'.format(checkpoint))    
    print('After being solved, each episode has:')
    print('-Avg score: {}\n-std: {}'.format(np.mean(scores[checkpoint:]),np.std(scores[checkpoint:])))
    
if __name__ == "__main__":
    filename = 'scores_DDQN_evaluation' 
    analyzeSingleRun(filename)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim([0,17])
    plt.axhline(y=13, linewidth = 0.5, color = 'black', linestyle = '--')
    
    filename = 'scores_DQN_train_C1250' 
    with open(filename,'rb') as f:
       scores, checkpoint = pickle.load(f)
       scores = scoresEvery100episodes(scores)
    plt.plot(np.arange(len(scores)), scores, linewidth = 0.8, label = 'DQN_c1250', color = 'blue')
    plt.axvline(x=checkpoint,ymax = 13/17,linewidth = 1, color = 'blue', linestyle = ':')
    
    filename = 'scores_DQN_train_C5000' 
    with open(filename,'rb') as f:
       scores, checkpoint = pickle.load(f)
       scores = scoresEvery100episodes(scores)
    plt.plot(np.arange(len(scores)), scores, linewidth = 0.8, label = 'DQN_c5000', color = 'green')
    plt.axvline(x=checkpoint, ymax = 13/17, linewidth = 1, color = 'green', linestyle = ':')
    
    filename = 'scores_DDQN_train_C1250' 
    with open(filename,'rb') as f:
       scores, checkpoint = pickle.load(f)
       scores = scoresEvery100episodes(scores)
    plt.plot(np.arange(len(scores)), scores, linewidth = 0.8, label = 'DDQN_c1250', color = 'orange')
    plt.axvline(x=checkpoint,ymax = 13/17, linewidth = 1, color = 'orange', linestyle = ':')
    
    filename = 'scores_DDQN_train_C5000' 
    with open(filename,'rb') as f:
       scores, checkpoint = pickle.load(f)
       scores = scoresEvery100episodes(scores)
    plt.plot(np.arange(len(scores)), scores, linewidth = 0.8, label = 'DDQN_c5000', color = 'brown')
    plt.axvline(x=checkpoint, ymax = 13/17, linewidth = 1, color = 'brown', linestyle = ':')
    
    plt.legend(loc = 'lower right')
    plt.title('Avg score obtained in 100 consecutive episodes')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    """