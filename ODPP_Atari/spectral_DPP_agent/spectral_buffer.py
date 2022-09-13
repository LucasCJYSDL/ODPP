import numpy as np


def discounted_sampling(ranges, discount):
    """Draw samples from the discounted distribution over 0, ...., n - 1, 
    where n is a range. The input ranges is a batch of such n`s.

    The discounted distribution is defined as
    p(y = i) = (1 - discount) * discount^i / (1 - discount^n).

    This function implement inverse sampling. We first draw
    seeds from uniform[0, 1) then pass them through the inverse cdf
    floor[ log(1 - (1 - discount^n) * seeds) / log(discount) ]
    to get the samples.
    """
    assert np.min(ranges) >= 1
    assert discount >= 0 and discount <= 1
    seeds = np.random.uniform(size=ranges.shape)
    if discount == 0:
        samples = np.zeros_like(seeds, dtype=np.int64)
    elif discount == 1:
        samples = np.floor(seeds * ranges).astype(np.int64)
    else:
        samples = (np.log(1 - (1 - np.power(discount, ranges)) * seeds) 
                / np.log(discount))
        samples = np.floor(samples).astype(np.int64)
    return samples


def uniform_sampling(ranges):
    return discounted_sampling(ranges, discount=1.0)


class EpisodicReplayBuffer:
    """Only store full episodes.
    
    Sampling returns EpisodicStep objects.
    """

    def __init__(self, max_size):
        self._max_size = max_size
        self._current_size = 0
        self._next_idx = 0
        self._episodes = []

    @property
    def current_size(self):
        return self._current_size

    @property
    def max_size(self):
        return self._max_size

    def add_steps(self, episode):
        """
        steps: a list of transitions: {'s': , 'a': , 'r': , 'next_s': , 'done': }.
        """
        if self._next_idx == self._current_size:
            if self._current_size < self._max_size:
                self._episodes.append(episode)
                self._current_size += 1
                self._next_idx += 1
            else:
                self._episodes[0] = episode
                self._next_idx = 1
        else:
            self._episodes[self._next_idx] = episode
            self._next_idx += 1

    def _sample_episodes(self, batch_size):
        return np.random.randint(self._current_size, size=batch_size)

    def _gather_episode_lengths(self, episode_indices):
        lengths = []
        for index in episode_indices:
            lengths.append(len(self._episodes[index]))
        return np.array(lengths, dtype=np.int64)

    def sample_steps(self, batch_size, mode='single', discount=None): # danger
        episode_indices = self._sample_episodes(batch_size)
        step_ranges = self._gather_episode_lengths(episode_indices)
        if mode=='single':
            step_indices = uniform_sampling(step_ranges)
            s = []
            for epi_idx, step_idx in zip(episode_indices, step_indices):
                s.append(self._episodes[epi_idx][step_idx]['s'])
            return s, None
        else:
            step1_indices = uniform_sampling(step_ranges - 1)
            if not discount:
                step2_indices = step1_indices + 1
            else:
                intervals = discounted_sampling(step_ranges - step1_indices - 1, discount=discount) + 1
                step2_indices = step1_indices + intervals
            s1 = []
            s2 = []
            for epi_idx, step1_idx, step2_idx in zip(episode_indices, step1_indices, step2_indices):
                s1.append(self._episodes[epi_idx][step1_idx]['s'])
                s2.append(self._episodes[epi_idx][step2_idx]['s'])
            return s1, s2

    def get_all_steps(self, max_num):
        if max_num < 0:
            max_num = 1e10  # collect all the data points

        s = []
        cur_len = 0
        for i in range(self._current_size):
            epi_len = len(self._episodes[i])
            for j in range(epi_len):
                s.append(self._episodes[i][j]['s'])
                cur_len += 1
            if cur_len >= max_num:  # the max_num can be exceeded
                break

        return s



