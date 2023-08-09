import gym

env_id = 'CartPole-v1'
env = gym.make(env_id)

print(env.action_space)
print(env.observation_space)
env.reset()
for _ in range(500):
    act = env.action_space.sample()
    obs, rwd, done, info = env.step(act)
    print(obs, rwd, done, info)
    # env.render()
    if done:
        break