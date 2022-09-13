import torch as th
import numpy as np
from types import SimpleNamespace as SN


class OneHot(object):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor): # check
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self):
        return (self.out_dim,), th.float32


class EpisodeBatch:
    def __init__(self, scheme, preprocess, batch_size, traj_length, data=None, device="cpu"):
        self.scheme = scheme.copy()
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.traj_length = traj_length
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {} # one for each time step
            self.data.episode_data = {} # one for an episode
            self._setup_data(self.scheme, self.preprocess, batch_size, traj_length)

    def _setup_data(self, scheme, preprocess, batch_size, traj_length):
        # preprocess
        for k in preprocess:
            assert k in scheme
            new_k = preprocess[k][0]
            transform = preprocess[k][1]
            vshape, dtype = transform.infer_output_info()
            self.scheme[new_k] = {"vshape": vshape, "dtype": dtype}
            if "traj_const" in self.scheme[k]:
                self.scheme[new_k]["traj_const"] = self.scheme[k]["traj_const"]

        # setup data
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            traj_const = field_info.get("traj_const", False)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)
            shape = vshape

            if traj_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, traj_length, *shape), dtype=dtype, device=self.device)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None)):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                v = self.preprocess[k][1].transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            is_first = True
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
                if is_first:
                    is_first = False
                    sample_size = new_data.transition_data[k].shape[0]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]
            ret = EpisodeBatch(self.scheme, preprocess={}, batch_size=sample_size, traj_length=self.traj_length, data=new_data, device=self.device)
            return ret

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        if isinstance(items, slice) or isinstance(items, (list, np.ndarray)): # [a, b, c]
            items = (items, slice(None))
        for item in items:
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices as they are
                parsed.append(item)
        return parsed

    def get_item(self, key):
        if key in self.data.transition_data:
            return self.data.transition_data[key]
        else:
            assert key in self.data.episode_data, key
            return self.data.episode_data[key]


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, preprocess, buffer_size, traj_length, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, preprocess, buffer_size, traj_length, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data, bs=slice(self.buffer_index, self.buffer_index + ep_batch.batch_size), ts=slice(0, ep_batch.traj_length))
            self.update(ep_batch.data.episode_data, bs=slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = self.buffer_index + ep_batch.batch_size
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size] #TODO: random shuffle
        else:
            # Uniform sampling
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]