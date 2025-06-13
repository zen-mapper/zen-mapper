import logging
from dataclasses import dataclass, field
from typing import List
import queue

import numpy as np
from scipy.stats import anderson
from sklearn.mixture import GaussianMixture
from zen_mapper.types import Cover

logger = logging.getLogger("zen_mapper")


@dataclass(order=True)
class QueueEntry:
    priority: float
    item: np.ndarray = field(compare=False)
    members: np.ndarray = field(compare=False)
    mask: np.ndarray = field(compare=False)
    n: int = field(compare=False)

    def __init__(self, interval, data):
        self.item = interval
        self.mask = (data >= self.item[0]) & (data <= self.item[1])
        self.members = data[self.mask]
        self.n = len(self.members)
        if self.n > 7:
            # the correction factor feels unnecessary
            self.priority = -anderson(self.members)[0]
        else:
            self.priority = 0


class GMapperCover:
    """
    An implementation of the G-mapper covering scheme.

    Parameters
    ----------
    iterations : int
        Maximum number of iterations for the algorithm
    max_intervals : int
        Maximum number of intervals to create
    method : str
        Method to use for splitting intervals ('DFS', 'BFS', or 'randomized')
    ad_threshold : float
        Threshold for Anderson-Darling test to determine when to split
    g_overlap : float
        Overlap factor for Gaussian mixture components

    Notes
        -----
        Be sure to cite: E. Alvarado, R. Belton, E. Fischer, K. J. Lee,
        S. Palande, S. Percival, and E. Purvine. (2025).
        G-mapper: Learning a cover in the mapper construction.
        SIAM Journal on Mathematics of Data Science, 7(2), 572-596.
    """

    def __init__(
        self,
        iterations=20,
        max_intervals=25,
        method="BFS",
        ad_threshold=5,
        g_overlap=0.3,
    ):
        self.iterations = iterations
        self.max_intervals = max_intervals
        self.method = method
        self.ad_threshold = ad_threshold
        self.g_overlap = g_overlap

    def __call__(self, data: np.ndarray) -> Cover:
        """
        Generate a cover using G-Mapper.

        Parameters
        ----------
        data : np.ndarray
            Input data to cover. The dataset (projection) should be 1D.

        Returns
        -------
        Cover
            A cover object that satisfies the Cover protocol
        """
        if len(data.shape) > 1 and data.shape[1] > 1:
            raise ValueError(
                f"data of shape {data.shape} is > 1D. The projection should be 1D."
            )
        else:
            lens = data

        initial_interval = [np.min(lens), np.max(lens)]
        return self._gmeans_algorithm(lens, initial_interval)

    def _gmeans_algorithm(self, lens, initial_interval):
        if self.method == "DFS":
            return self._dfs(lens, initial_interval)
        elif self.method == "BFS":
            return self._bfs(lens, initial_interval)
        elif self.method == "randomized":
            return self._randomized(lens, initial_interval)

        raise ValueError(f"Unknown method {self.method}")

    def _randomized(self, lens, intervals):
        raise NotImplementedError(
            "'randomized' option is not available. Try BFS instead."
        )

    def _bfs(self, lens, initial_interval):
        q = queue.PriorityQueue()
        initial_interval_entry = QueueEntry(initial_interval, lens)
        q.put(initial_interval_entry)

        cover = []
        iteration_count = 0

        while (
            not q.empty()
            and iteration_count < self.iterations
            and len(cover) < self.max_intervals
        ):
            large_ad_element = q.get()

            if large_ad_element.priority > -self.ad_threshold or large_ad_element.n < 8:
                cover.append(np.where(np.isin(lens, large_ad_element.members))[0])
            else:
                split_intervals = _gm_split(
                    large_ad_element.item, large_ad_element.members, self.g_overlap
                )
                new_entry_1 = QueueEntry(split_intervals[0], lens)
                new_entry_2 = QueueEntry(split_intervals[1], lens)
                q.put(new_entry_1)
                q.put(new_entry_2)

            iteration_count += 1

        while not q.empty():
            remaining_element = q.get()
            print(remaining_element.members)
            cover.append(np.where(np.isin(lens, remaining_element.members))[0])

        return cover

    def _dfs(self, lens, intervals):
        raise NotImplementedError("'DFS' option is not available. Try BFS instead.")


def _gm_split(interval, membership_data, g_overlap):
    if len(membership_data) == 0:
        mid = (interval[0] + interval[1]) / 2
        return np.array([[interval[0], mid], [mid, interval[1]]])

    c = np.mean(membership_data)
    std = np.std(membership_data, ddof=1)

    if std == 0:
        mid = (interval[0] + interval[1]) / 2
        return np.array([[interval[0], mid], [mid, interval[1]]])

    m = np.sqrt(2 / np.pi) * std
    c1 = c + m
    c2 = c - m

    L = np.array(membership_data).reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=2, means_init=[[c1], [c2]], covariance_type="full"
        ).fit(L)

        left_index = np.argmin(gmm.means_)
        left_mean = np.min(gmm.means_)
        left_std = np.sqrt(gmm.covariances_[left_index])[0][0]

        right_index = np.argmax(gmm.means_)
        right_mean = np.max(gmm.means_)
        right_std = np.sqrt(gmm.covariances_[right_index])[0][0]

        # calculate split point and overlap
        split_factor = (1 + g_overlap) * left_std / (left_std + right_std)
        split_point = left_mean + split_factor * (right_mean - left_mean)
        left_interval = [interval[0], min(split_point, interval[1])]
        right_interval = [
            max(
                interval[0],
                right_mean
                - (1 + g_overlap)
                * right_std
                / (left_std + right_std)
                * (right_mean - left_mean),
            ),
            interval[1],
        ]

        return np.array([left_interval, right_interval])

    except Exception as e:
        logger.warning(f"GMM fitting failed: {e}. Falling back to simple split...")
        mid = (interval[0] + interval[1]) / 2
        return np.array([[interval[0], mid], [mid, interval[1]]])
