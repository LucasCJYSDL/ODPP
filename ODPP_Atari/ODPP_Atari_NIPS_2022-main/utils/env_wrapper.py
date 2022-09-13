from multiprocessing import Pipe, Process
from functools import partial
import numpy as np
import gym

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            action = data
            # Take a step in the environment
            if np.square(action).sum() >= 1e4:
                print("danger: ", action)
                action = env.action_space.sample()
            next_s, reward, done, info = env.step(action)
            remote.send({"next_state": next_s, "reward": reward, "done": done, "info": info})
        elif cmd == "reset":
            init_s = env.reset()
            remote.send({"state": init_s})
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "set_seed":
            seed = data
            env.seed(seed)
        elif cmd == "set_init_state":
            state = data
            env.set_init_state(state)
        else:
            raise NotImplementedError

class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def env_fn(env_id):
    return gym.make(env_id)

class EnvWrapper(object):
    def __init__(self, env_id, seed, env_num):
        self.env_num = env_num

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.env_num)])
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, env_id)))) for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        for idx in range(self.env_num):
            temp_seed = seed + idx + 1 # to be different from the 'env' in main.py
            self.parent_conns[idx].send(('set_seed', temp_seed))

    def close(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        init_states = []
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            init_states.append(data["state"])

        return np.array(init_states)

    def step(self, action_array, done_vec, s):
        for idx, parent_conn in enumerate(self.parent_conns):
            if not done_vec[idx]:
                parent_conn.send(("step", action_array[idx]))

        next_s = np.zeros_like(s, dtype=np.float32)
        r = np.zeros((self.env_num, ), dtype=np.float32)
        fwd_r = np.zeros((self.env_num, ), dtype=np.float32)
        sde_r = np.zeros((self.env_num, ), dtype=np.float32)
        ctr_r = np.zeros((self.env_num, ), dtype=np.float32)
        sur_r = np.zeros((self.env_num, ), dtype=np.float32)
        done = [True for _ in range(self.env_num)]

        for idx, parent_conn in enumerate(self.parent_conns):
            if not done_vec[idx]:
                data = parent_conn.recv()
                next_s[idx] = data['next_state']
                r[idx] = data['reward']
                done[idx] = data['done']

        return next_s, r, done, fwd_r, sde_r, ctr_r, sur_r

    def set_init_state(self, s):
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("set_init_state", s[idx]))


if __name__ == '__main__':
    test = EnvWrapper('MountainCarContinuous-v0', 0, 5)
    print(test.reset())