# Reinforcement Learning Agents
This repository contains implementations of Agents using [Nervana Systems Coach](https://github.com/NervanaSystems/coach), and bayesian optimization implementations using [Sheffields GPyOpt](https://github.com/SheffieldML/GPyOpt).

These scripts require that you use Linux. 

There is also a heavy reliance on the use of Pandas to mange .csv files for logging data during training, this is implemented in such a way that while running Bayesian Optimization if for any reason the environment/agent crashes the failed log files will be removed.

This repository is realized quite quickly and not everything has been extensively tested. However the premis behind the code works, Bayesian Optimization of RL Coach Agents for any arbitrary environment considering they make use of either OpenAI Gym or the RL Coach interface.

## How it works:
### In theory:
Find optimal hyperparameters using the following equation:

<img src="./images/equations/hyperparameter_tuning.svg">

f(x) is defined as the sum of 'Total rewards' for all episodes in a training cycle. 
Here are some reasons for this:
- Very Easy to implement
- Optimizes for faster learning
- Optimizes for better/more stable learning
- Optimizes for maximum reward

Of course there is a possibility that a local-maximum is found, but in my experience this is sufficient in achieving acceptable results.

### How it is implemented: 
Using the library GPyOpt, a gaussian process bayesian optimizer is constructed the two mandatory params that must be passed are:

1. Boundary definitions of the hyper-parameter set <img src="./images/equations/X.svg"> as a list of dicts.

2. The function <img src="./images/equations/func.svg"> can be implemented in python as an algorithm/function such as def run_ai(x): do stuff; return y

The acquisition funtion for determining the next choice of hyper-parameters is the Expected Improvement function by default.

#### In practice:
The algorithm/function defined here actually performs 3 steps:
1. It writes the new parameters to an opt_params.csv file
2. It calls the Agent .py script and waits for this code to finish running.
3. After correct execution of the Agent .py script reads the log-file and sums + returns the total-reward for each episode. If the agent script crashes this code deletes the last training data and exits the script as well.

#### The Agent .py script:
In the Agent script the opt_params.csv file is read and the latest hyper-parameters entry is used to construct a new agent.

The new agent is trained for a predefined number of iterations and upon completion the hyperparameter optimization process is resumed. 

#
The output of the bayesopt.py script is the optimization_parameters.csv file. 
Agents implemented in Reinforcement Learning Coach automatically generate log files that are used for both the Dashboard app that comes with RL Coach as well as the Optimization script implemented in this Repo. 

This means that any Agent realized in RL Coach can easily be optimzed using bayesopt.py all that is required is having the Agent load new parameters from the opt_params.csv file and defining the boundaries of the hyper-parameter searchspace.

Goals for in the future if i have time:
- Implement multi-agent optimization techniques
- Pass RL algorithms themselves as a hyper-parameter in an attempt to perform AutoRL.
- Implement CEM and Particle Swarm Optimization as alternate optimization algorithms.