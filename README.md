# Reinforcement Learning Agents
This repository contains implementations of Agents using [Nervana Systems Coach](https://github.com/NervanaSystems/coach), and bayesian optimization implementations using [Sheffields GPyOpt](https://github.com/SheffieldML/GPyOpt).

These scripts require that you use Linux. 

There is also a heavy reliance on the use of Pandas to mange .csv files for logging data during training, this is implemented in such a way that while running Bayesian Optimization if for any reason the environment/agent crashes the failed log files will be removed.

This repository is realized quite quickly and not everything has been extensively tested. However the premis behind the code works, Bayesian Optimization of RL Coach Agents for any arbitrary environment considering they make use of either OpenAI Gym or the RL Coach interface.

How it works:
Find optimal hyperparameters using the following equation:

<img src="./images/equations/hyperparameter_tuning.svg">

f(x) is defined as the sum of 'Total rewards' for all episodes in a training cycle. 
Here are some reasons for this:
- Very Easy to implement
- Optimizes for faster learning
- Optimizes for better/more stable learning
- Optimizes for maximum reward

Of course there is a possibility that a local-maximum is found, but in my experience this is sufficient in achieving acceptable results.

