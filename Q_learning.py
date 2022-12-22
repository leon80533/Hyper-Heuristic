from hh_env import HHEnv
import math
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time



class HHagent():
    def __init__(self, problem_domain = 'SAT',
                 instance = 3, time = 100, min_lr = 0.3, 
                 min_epsilon = 0.1, discount = 0.5, decay = 25,  unimprovement_time = 40,
                 is_backtrack = False, random_store = False, is_restart = False, res_q = False):
        
        self.problem_domain = problem_domain
        self.env = HHEnv(self.problem_domain)
        self.instance = instance
        self.env.initialize(self.instance)        
        self.buckets = (self.env.action_space.n + 1, self.env.tail_length + 1)
        
        self.time = time
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.unimprovement_time = unimprovement_time
        self.is_backtrack = is_backtrack
        self.random_store = random_store
        self.is_restart = is_restart
        self.res_q = res_q
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        self.funV = []
        self.functionValue = []
        self.Bests = []
        
        
    def discretize_state(self, obs):
        discretized = list()
        for i in range(len(obs)):
            discretized.append(int(obs[i]))

        return tuple(discretized)

    def choose_action(self, state):
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample() 
        else:
            return np.argmax(self.Q_table[state])

    def update_q(self, state, action, reward, new_state):
        self.Q_table[state][action] += (self.learning_rate * 
                                        (reward 
                                         + self.discount * np.max(self.Q_table[new_state]) 
                                         - self.Q_table[state][action]))
        

    def get_epsilon(self, t):
        """Gets value for epsilon. It declines as we advance in episodes."""
        # Ensures that there's almost at least a min_epsilon chance of randomly exploring
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        """Gets value for learning rate. It declines as we advance in episodes."""
        # Learning rate also declines as we add more episodes
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))
    
    
    def train(self):
        e = -1
        overall_time = 0
        start_time = time.time()
        best = np.inf
        best_time = start_time
        best_of_all_time = np.inf
        
        while self.time > overall_time:
            e += 1
            end_time = time.time()
            overall_time = end_time - start_time
            
            # reset or intiailize the state
            if end_time - best_time > self.unimprovement_time:
                if self.is_restart:
                    # print("+++++++++UPDATE_RESTART+++++++++++++++++")
                    self.Bests.append(self.env.problem.getBestSolutionValue())
                    best = np.inf
                    self.env = HHEnv(self.problem_domain)
                    current_state = self.discretize_state(self.env.initialize(self.instance))
                    if self.res_q:
                        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
                if self.is_backtrack:
                    # print("+++++++++Go to Random Best+++++++++++++++++")
                    self.Bests.append(self.env.problem.getBestSolutionValue())
                    self.env.start_from_previous()
                    best = np.inf
                    current_state = self.discretize_state(self.env.reset())
                else:
                    self.Bests.append(self.env.problem.getBestSolutionValue())
                    current_state = self.discretize_state(self.env.reset())
            else:
                self.Bests.append(self.env.problem.getBestSolutionValue())
                current_state = self.discretize_state(self.env.reset())

            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False
            
            # Looping for each step
            while not done:
                # Choose A from S
                action = self.choose_action(current_state)
                
                # Take action
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(obs)
                # Update Q(S,A)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state
                
                # We break out of the loop when done is False which is
                # a terminal state.
            # if e % 100 == 0:
            #     print("Current Funciton Value: {}".format(self.env.problem.getFunctionValue(0)))
            
            if self.env.problem.getFunctionValue(0) < best:
    
                best = self.env.problem.getFunctionValue(0)
                best_time = end_time
                if self.is_backtrack:
                    if self.random_store:
                        random_store = random.uniform(0, 1)
                        if random_store < 0.3:
                            self.env.store_best()
                    self.env.store_best()
            
            if best_of_all_time > self.env.problem.getBestSolutionValue():
                best_of_all_time = self.env.problem.getBestSolutionValue()
                
            self.funV.append(self.env.problem.getFunctionValue(0))
        # print('Finished training! Episode is {}'.format(e))
        # print("Recorded best solutnion is {}".format(best_of_all_time))
    
        return min(self.Bests)
    
    def plot_learning(self):
        """
        Plots the number of steps at each episode and prints the
        amount of times that an episode was successfully completed.
        """
        sns.lineplot(x = range(len(self.funV)), y = self.funV)
        
    def save(self):
        name = self.problem_domain + "_" + str(self.instance) + "_Qtable"
        if self.is_backtrack:
            name += '_back'
            if self.random_store:
                name += '_rand'
        if self.is_restart:
            name += "_restart"
            if self.res_q:
                name += "_resq"
        location = "q_tables/" + name
        np.save(location,self.Q_table)
        
        
def run_all(train_time):
    all_instances = {"SAT": [3, 5, 4, 10, 11], 'BP':[7, 1, 9, 10, 11], 'TSP': [0, 8, 2, 7, 6]}
    best_org_sat = []
    best_restart_sat = []
    best_restart_q_sat = []
    best_backtrack_sat = []
    best_backtrack_random_sat = []
    best_org_bin = []
    best_restart_bin = []
    best_restart_q_bin = []
    best_backtrack_bin = []
    best_backtrack_random_bin = []
    best_org_tsp = []
    best_restart_tsp = []
    best_restart_q_tsp = []
    best_backtrack_tsp = []
    best_backtrack_random_tsp = []
    
    for i in all_instances.keys():
        for j in all_instances[i]:
            
            for h in range(5):
                agent = HHagent(problem_domain=i,instance = j,time = train_time)
                print("Training orginal")
                best = agent.train()
                if i == "SAT":
                    best_org_sat.append(best)
                elif i == 'BP':
                    best_org_bin.append(best)
                else:
                    best_org_tsp.append(best)
                agent.plot_learning()
                agent.save()
                
            for h in range(5):
                agent = HHagent(problem_domain = i, instance = j, time = train_time, is_restart = True)
                print('Training Restart')
                best = agent.train()
                if i == "SAT":
                    best_restart_sat.append(best)
                elif i == 'BP':
                    best_restart_bin.append(best)
                else:
                    best_restart_tsp.append(best)
                agent.plot_learning()
                agent.save()
                
            for h in range(5):
                agent = HHagent(problem_domain = i, instance = j, time = train_time, is_restart = True, res_q = True)
                print('Training Restart and Q')
                best = agent.train()
                if i == "SAT":
                    best_restart_q_sat.append(best)
                elif i == 'BP':
                    best_restart_q_bin.append(best)
                else:
                    best_restart_q_tsp.append(best)
                agent.plot_learning()
                agent.save()
                
            for h in range(5):
                agent = HHagent(problem_domain = i, instance = j, time = train_time, is_backtrack = True)
                print("Training Backtrack")
                best = agent.train()
                if i == "SAT":
                    best_backtrack_sat.append(best)
                elif i == 'BP':
                    best_backtrack_bin.append(best)
                else:
                    best_backtrack_tsp.append(best)
                agent.plot_learning()
                agent.save()

            for h in range(5):
                agent = HHagent(problem_domain = i, instance = j, time = train_time, is_backtrack = True, random_store = True)
                print("Training BackTrack with Random")
                best = agent.train()
                if i == "SAT":
                    best_backtrack_random_sat.append(best)
                elif i == 'BP':
                    best_backtrack_random_bin.append(best)
                else:
                    best_backtrack_random_tsp.append(best)
                agent.plot_learning() 
                agent.save()
    
    results = {"SAT_result":[best_org_sat,best_restart_sat,best_restart_q_sat,best_backtrack_sat,best_backtrack_random_sat],
               "BP_result":[best_org_bin,best_restart_bin,best_restart_q_bin,best_backtrack_bin,best_backtrack_random_bin],
               "TSP_result":[best_org_tsp,best_restart_tsp,best_restart_q_tsp,best_backtrack_tsp,best_backtrack_random_tsp]}    
            
    
    return results
        

if __name__ == "__main__":
    reult = run_all(600)
    