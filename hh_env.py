import gym
import py4j
from py4j.java_gateway import JavaGateway, GatewayParameters
import random
import time
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import math
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


class HHEnv(gym.Env):
    def __init__(self, problem_domain='TSP', time_limit=600, lubylength=200000, last_h=9, tail_length=10):
        gateway = JavaGateway(
            gateway_parameters=GatewayParameters(address=u'10.200.77.92'))

        hyflex = gateway.jvm
        self.action_space = None
        self.problem_domain = problem_domain
        if problem_domain == 'SAT':
            self.problem = hyflex.SAT.SAT(1234)
        elif problem_domain == 'BP':
            self.problem = hyflex.BinPacking.BinPacking(1234)
        elif problem_domain == 'TSP':
            self.problem = hyflex.travelingSalesmanProblem.TSP(1234)

        self.last_h = last_h
        self.tail_length = tail_length

        low = np.array(
            [-1,
                -1
             ],
            dtype=np.float32,
        )
        high = np.array(
            [self.last_h,
                self.tail_length
             ],
            dtype=np.float32,
        )
        self.observation_space = Box(low, high)
        self.state = None
        self.done = False
        self.solution = None
        self.h_chain = None
        self.luby = None
        self.episode = 0
        self.lubylength = lubylength
        self.origin_sol = None
        self.time_limit = time_limit
        self.current_time = None
        self.overall_time = 0
        self.exceed = False
        self.totalDelta = 0
        self.sidewaysMoveFound = False
        self.costCurrent = None

        def luby(length=200000):
            L = [-1]*length
            for i in range(1, length+1):
                #         print("current index: ",i)
                left = math.log2(i + 1)
                right = int(left)
                k = right
                if left == right:
                    L[i-1] = 2 ** (k-1)
                else:
                    k += 1
                    L[i-1] = L[i-1 - 2**(k-1) + 1]
            return L

        self.luby = luby(length=self.lubylength)

    def step(self, action):

        # setting time
        t1 = time.time()

        # last heuristic applied
        last_heu = self.h_chain[self.iter_num-1]

        # update hyper-heuristic chain
        self.h_chain[self.iter_num] = int(action)

        self.iter_num += 1

        # number of heuristic left if bigger than max tail length it equal max tail length
        num_heur_left = len(self.h_chain) - self.iter_num

        num_heur_left = min(self.tail_length, num_heur_left)

        ori = self.origin_sol

        self.origin_sol = self.problem.applyHeuristic(int(action), 1, 1)
        delta = self.origin_sol - ori

        t2 = time.time()

        self.current_time += t2-t1
        self.overall_time += t2-t1
        self.totalDelta += delta

        end = True if (len(self.h_chain)) == self.iter_num else False

        if end:
            self.done = True

        if self.totalDelta < 0:
            self.current_time = max(1, self.current_time)
            reward = (- self.totalDelta)/(self.current_time)
        else:
            reward = 0

        self.state = (last_heu, num_heur_left)

        info = {}
        # Return step information
        self.solution = self.problem.getBestSolutionValue()
        if self.overall_time > self.time_limit:
            self.exceed == True
        if self.totalDelta < 0:
            self.problem.copySolution(1, 0)
            self.done = True
        elif self.totalDelta == 0:
            self.problem.copySolution(1, 2)
            self.sidewaysMoveFound = True
        elif end and self.sidewaysMoveFound and self.totalDelta >= 0:
            self.problem.copySolution(2, 0)

        return np.array(self.state, dtype=np.float32), reward, self.done, info

    def render(self):
        # Implement viz
        pass

    def initialize(self, instance):
        self.problem.loadInstance(instance)
        self.problem.setMemorySize(4)
        self.problem.initialiseSolution(0)
        self.problem.copySolution(0, 1)
        self.problem.copySolution(0, 3)
        self.origin_sol = self.problem.getFunctionValue(0)
        self.action_space = Discrete(self.problem.getNumberOfHeuristics())
        self.iter_num = 0
        self.current_time = 0
        self.h_chain = [-1 for i in range(self.luby[self.episode])]
        self.episode += 1
        self.state = (-1, min(10, len(self.h_chain)))
        self.sidewaysMoveFound = False
        self.done = False
        return np.array(self.state, dtype=np.float32)

    def reset(self):
        self.origin_sol = self.problem.getBestSolutionValue()
        self.problem.copySolution(0, 1)
        self.sidewaysMoveFound = False
        self.iter_num = 0
        self.current_time = 0
        self.totalDelta = 0
        self.h_chain = [-1 for i in range(self.luby[self.episode])]
        self.episode += 1
        self.state = (-1, min(10, len(self.h_chain)))
        self.done = False
        return np.array(self.state, dtype=np.float32)

    def start_from_previous(self):
        self.problem.copySolution(3, 1)

    def store_best(self):
        self.problem.copySolution(0, 3)
