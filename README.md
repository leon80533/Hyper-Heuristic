# HyFlex

HyFlex (Hyper-heuristics Flexible framework) is a Java object-oriented framework used to implement and compare different iterative generic heuristics search algorithms (also known as Hyper-heuristics).

See the HyFlex web site for details: http://www.asap.cs.nott.ac.uk/external/chesc2011 - web archive

The goal of this project is to collect hyper-heurisics from CHeSC 2011 (Cross-domain Heuristic Search Challenge) and enable to reproduce the results of the challenge.

Hyper-heuristic implementers might find this environment helpful for comparing own results with other approaches.

# Hyper-Heuristic

**hh_env.py**  initialized for the environment of the RL.

**Q_Learning.py** the algorithmn used and result

hyflex_python.jar should first be executed to start the py4j server and connect to the hyflex jar. There will be a server url address generated.
Copy the URL address in the hh_env.py to intialize the environment. 
Run Q_learning.py to see the result.

