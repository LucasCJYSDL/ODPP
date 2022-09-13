# Evaluation of Unsupervised Option Discovery based on Determinant Point Process on Atari Games 

## How to config the environments:
- On Ubuntu 18.04
- python 3.6
- pytorch 1.6
- tensorboard 2.5
- atari_py 0.2.6
- gym  0.19.0
- matplotlib
- tqdm
- seaborn
- ...


## How to run the experiments on the Atari tasks

- To run the code with algorithm XXX (i.e., VIC, VALOR, DIAYN, ODPP or DCO) and random seed Y for which we simply choose 0, 1, or 2, on task ZZZ (i.e., AirRaid-ram-v0, CartPole-v1, or Riverraid-ram-v0):

```bash
python main.py --agent_id='XXX' --seed=Y ----env_id='ZZZ'
```

- For the simpler task 'CartPole-v1', we learn options with a longer horizon, i.e., 100. Hence, you need to change the value of 'args.traj_length' on Line 29 of 'configs.py' to 100. While, for the other tasks, the value of 'args.traj_length' should be 50.
- The tensorboard files containing the training information can be found in 'option_agent/log', within which the term 'tot_rwd' records the change of the trajectory reward in the training process.
- The introduction of the Atari environments that we have evaluated on can be found in [Atari Wiki](https://github.com/openai/gym/wiki/Table-of-environments).