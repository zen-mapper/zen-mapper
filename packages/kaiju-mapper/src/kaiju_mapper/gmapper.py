import logging
import sys
from typing import List

import numpy as np
from scipy.stats import anderson
from sklearn.mixture import GaussianMixture
from zen_mapper.types import Cover

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
logger = logging.getLogger("zen_mapper")


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
            Input data to cover. If the dataset is not 1D then it will project
            to the first axis.

        Returns
        -------
        Cover
            A cover object that satisfies the Cover protocol
        """
        if len(data.shape) > 1 and data.shape[1] > 1:
            lens = data[:, 0]
        else:
            lens = data

        initial_intervals = np.array([[np.min(lens), np.max(lens)]])
        result_intervals = self._gmeans_algorithm(lens, initial_intervals)

        return compute_covers(result_intervals, data)

    def _gmeans_algorithm(self, lens, initial_intervals):
        intervals = initial_intervals.copy()
        check_interval = [True for _ in range(len(intervals))]
        interval_membership = _membership(lens, intervals)
        ad_scores = []
        split_index = []

        for iteration in range(self.iterations):
            modified = False

            if self.method is None or self.method == "DFS":
                for i in range(len(intervals)):
                    if not check_interval[i]:
                        continue

                    if len(interval_membership[i]) == 0:
                        check_interval[i] = False
                        continue

                    if ad_test(interval_membership[i]) > self.ad_threshold:
                        check_interval[i] = True
                        check_interval.insert(i + 1, True)
                        tem = len(interval_membership[i])

                        new_intervals, interval_membership = _split(
                            interval_membership, intervals, self.g_overlap, i
                        )
                        intervals = new_intervals

                        if tem == len(interval_membership[i]):
                            check_interval[i] = False
                            continue
                        if tem == len(interval_membership[i + 1]):
                            check_interval[i + 1] = False
                            continue

                        ad_scores = [
                            ad_test(interval_membership[i]),
                            ad_test(interval_membership[i + 1]),
                        ]

                        if ad_scores[1] > ad_scores[0]:
                            temp = intervals[i + 1].copy()
                            intervals = np.delete(intervals, i + 1, axis=0)
                            intervals = np.insert(intervals, i, [temp], axis=0)

                            temp = interval_membership[i + 1]
                            interval_membership.pop(i + 1)
                            interval_membership.insert(i, temp)

                        modified = True
                        break
                    else:
                        check_interval[i] = False

                if not modified:
                    logger.info(f"Convergence after {iteration} iterations.")
                    return intervals

            elif self.method == "BFS":
                if len(ad_scores) == 0:
                    for i in range(len(intervals)):
                        if not check_interval[i]:
                            continue

                        if len(interval_membership[i]) == 0:
                            check_interval[i] = False
                            continue

                        if ad_test(interval_membership[i]) > self.ad_threshold:
                            split_index.append(i)
                            ad_scores.append(ad_test(interval_membership[i]))
                            modified = True
                        else:
                            check_interval[i] = False

                    if not modified:
                        logger.info(f"Convergence after {iteration} iterations.")
                        return intervals

                    ad_scores = [0 if x != x else x for x in ad_scores]  # Handle NaN

                    if max(ad_scores) == 0:
                        logger.info(f"Convergence after {iteration} iterations.")
                        return intervals

                    best_split = ad_scores.index(max(ad_scores))
                    j = split_index[best_split]
                    check_interval[j] = True
                    check_interval.insert(j + 1, True)

                    new_intervals, interval_membership = _split(
                        interval_membership, intervals, self.g_overlap, j
                    )
                    intervals = new_intervals

                    del ad_scores[best_split]
                    del split_index[best_split]

                else:
                    for i in range(len(intervals)):
                        if not check_interval[i]:
                            continue

                        if len(interval_membership[i]) == 0:
                            check_interval[i] = True
                            continue

                        if ad_test(interval_membership[i]) > self.ad_threshold:
                            modified = True
                        else:
                            check_interval[i] = False

                    if not modified:
                        logger.info(f"Convergence after {iteration} iterations.")
                        return intervals

                    ad_scores = [0 if x != x else x for x in ad_scores]  # Handle NaN

                    if max(ad_scores) == 0:
                        logger.info(f"Convergence after {iteration} iterations.")
                        return intervals

                    best_split = ad_scores.index(max(ad_scores))
                    j = split_index[best_split]
                    check_interval[j] = True
                    check_interval.insert(j + 1, True)

                    new_intervals, interval_membership = _split(
                        interval_membership, intervals, self.g_overlap, j
                    )
                    intervals = new_intervals

                    del ad_scores[best_split]
                    del split_index[best_split]

            elif self.method == "randomized":
                all_elements_idx = [i for i in range(len(intervals))]
                element_ad_scores = [
                    ad_test(interval_membership[i]) for i in range(len(intervals))
                ]
                element_ad_scores = [
                    0 if x != x else x for x in element_ad_scores
                ]  # Handle NaN

                if sum(element_ad_scores) == 0:
                    logger.info(
                        f"Convergence after {iteration} iterations - all AD scores are zero."
                    )
                    return intervals

                found_valid = False
                while not found_valid and len(all_elements_idx) > 0:
                    # Sample one of the intervals weighted by ad score
                    weights = np.asarray(element_ad_scores)[all_elements_idx]
                    if weights.sum() == 0:
                        # If all weights are zero, use uniform weights
                        weights = np.ones_like(weights) / len(weights)
                    else:
                        weights = weights / weights.sum()

                    current_element = int(
                        np.random.choice(np.asarray(all_elements_idx), p=weights)
                    )
                    j = current_element

                    if len(interval_membership[j]) == 0:
                        removal_idx = all_elements_idx.index(j)
                        all_elements_idx.pop(removal_idx)
                        continue

                    if ad_test(interval_membership[j]) > self.ad_threshold:
                        check_interval[j] = True
                        check_interval.insert(j + 1, True)

                        new_intervals, interval_membership = _split(
                            interval_membership, intervals, self.g_overlap, j
                        )
                        intervals = new_intervals

                        found_valid = True
                    else:
                        removal_idx = all_elements_idx.index(j)
                        all_elements_idx.pop(removal_idx)

                if not found_valid:
                    logger.info(
                        f"Convergence after {iteration} iterations - no valid splits found."
                    )
                    return intervals

            if len(intervals) > self.max_intervals:
                logger.info(
                    f"Reached maximum number of intervals ({self.max_intervals})."
                )
                break

        return intervals


def _split(interval_membership, intervals, g_overlap, index):
    j = index
    split_interval = _gm_split(
        intervals[j], np.array(interval_membership[j]), g_overlap
    )

    new_intervals = np.delete(intervals, j, axis=0)
    new_intervals = np.insert(new_intervals, j, split_interval, axis=0)

    new_membership = _membership(interval_membership[j], split_interval)
    interval_membership.pop(j)
    interval_membership.insert(j, new_membership[0])
    interval_membership.insert(j + 1, new_membership[1])

    return new_intervals, interval_membership


def _membership(data, intervals):
    """
    Assign data points to intervals.

    Returns
    -------
    list
        List of lists containing data points for each interval
    """
    return [
        [x for x in data if interval[0] <= x <= interval[1]] for interval in intervals
    ]


def ad_test(data):
    """
    Anderson-Darling Test for normality.

    Parameters
    ----------
    data : list or np.ndarray
        Data points to test for normality

    Returns
    -------
    float
        Anderson-Darling test statistic (corrected)
    """
    n = len(data)

    if n == 0:
        return 0
    try:
        and_corrected = anderson(data)[0] * (1 + 4 / n - 25 / (n**2))
        return and_corrected
    except Exception as e:
        logger.warning(f"Anderson-Darling test failed: {e}")
        return 0


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


def compute_covers(intervals: np.ndarray, data: np.ndarray) -> List[np.ndarray]:
    if len(data.shape) > 1 and data.shape[1] > 1:
        lens = data[:, 0]
    else:
        lens = data.flatten()

    return [
        np.where((lens >= interval[0]) & (lens <= interval[1]))[0]
        for interval in intervals
    ]
