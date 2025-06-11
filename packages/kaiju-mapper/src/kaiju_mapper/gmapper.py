import logging
import operator
from typing import List

import numpy as np
from scipy.stats import anderson
from sklearn.mixture import GaussianMixture
from zen_mapper.types import Cover

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
            Input data to cover. The dataset (projection) should be 1D.

        Returns
        -------
        Cover
            A cover object that satisfies the Cover protocol
        """
        if len(data.shape) > 1 and data.shape[1] > 1:
            raise ValueError(
                f"data of shape {len(data.shape)} is > 1. The projection should be 1D."
            )
        else:
            lens = data

        initial_intervals = np.array([[np.min(lens), np.max(lens)]])
        result_intervals = self._gmeans_algorithm(lens, initial_intervals)

        return [
            np.where((lens >= interval[0]) & (lens <= interval[1]))[0]
            for interval in result_intervals
        ]

    def _gmeans_algorithm(self, lens, initial_intervals):
        if self.method == "DFS":
            return self._dfs(lens, initial_intervals)
        elif self.method == "BFS":
            return self._bfs(lens, initial_intervals)
        elif self.method == "randomized":
            return self._randomized(lens, initial_intervals)

        raise ValueError(f"Unknown method {self.method}")

    def _randomized(self, lens, intervals):
        interval_membership = _membership(lens, intervals)
        for iteration in range(self.iterations):
            if len(intervals) >= self.max_intervals:
                logger.info(
                    f"Reached maximum number of intervals ({self.max_intervals})."
                )
                break

            all_elements_idx = np.ones(len(intervals), dtype=bool)
            element_ad_scores = [
                ad_test(interval_membership[i]) for i in range(len(intervals))
            ]
            element_ad_scores = [
                0 if np.isnan(x) else x for x in element_ad_scores
            ]  # Handle NaN

            if sum(element_ad_scores) == 0:
                logger.info(
                    f"Convergence after {iteration} iterations - all AD scores are zero."
                )
                break

            found_valid = False
            while not found_valid and np.any(all_elements_idx):
                # Sample one of the intervals weighted by ad score
                weights = np.asarray(element_ad_scores)[all_elements_idx]

                if weights.sum() == 0:
                    weights = None
                else:
                    weights /= weights.sum()

                current_element = np.random.choice(
                    np.asarray(np.flatnonzero(all_elements_idx), dtype=int),
                    p=weights,
                )

                if len(interval_membership[current_element]) == 0:
                    all_elements_idx[current_element] = False
                    continue

                if ad_test(interval_membership[current_element]) > self.ad_threshold:

                    new_intervals, interval_membership = _split(
                        interval_membership, intervals, self.g_overlap, current_element
                    )
                    intervals = new_intervals

                    found_valid = True
                else:
                    all_elements_idx[current_element] = False

            if not found_valid:
                logger.info(
                    f"Convergence after {iteration} iterations - no valid splits found."
                )
                break

        return intervals

    def _bfs(self, lens, intervals):
        check_interval = set(range(len(intervals)))
        interval_membership = _membership(lens, intervals)
        ad_scores: dict[int, float] = dict()

        for iteration in range(self.iterations):
            if len(intervals) >= self.max_intervals:
                logger.info(
                    f"Reached maximum number of intervals ({self.max_intervals})."
                )
                break

            append_scores = len(ad_scores) == 0
            intervals_to_remove = []

            # copy to avoid runtime error
            intervals_to_check = list(check_interval)

            for i in intervals_to_check:
                test_result = ad_test(interval_membership[i])

                if test_result <= self.ad_threshold:
                    intervals_to_remove.append(i)
                    continue

                if append_scores:
                    ad_scores[i] = test_result

            # remove intervals AFTER iteration
            for i in intervals_to_remove:
                check_interval.discard(i)

            if not check_interval:
                logger.info(f"Convergence after {iteration} iterations.")
                break

            best_split, score = max(ad_scores.items(), key=operator.itemgetter(1))

            if score == 0:
                logger.info(f"Convergence after {iteration} iterations.")
                break

            intervals, interval_membership = _split(
                interval_membership, intervals, self.g_overlap, best_split
            )

            # update after split
            check_interval.add(best_split)
            check_interval.add(best_split + 1)

            # account for shifted indices
            new_ad_scores = {}
            for idx, score_val in ad_scores.items():
                if idx == best_split:
                    continue
                elif idx > best_split:
                    new_ad_scores[idx + 1] = score_val
                else:
                    new_ad_scores[idx] = score_val
            ad_scores = new_ad_scores

        return intervals

    def _dfs(self, lens, intervals):
        check_interval = set(range(len(intervals)))
        interval_membership = _membership(lens, intervals)

        for iteration in range(self.iterations):
            if len(intervals) >= self.max_intervals:
                logger.info(
                    f"Reached maximum number of intervals ({self.max_intervals})."
                )
                break

            to_split = None
            # copy to avoid runtime error
            intervals_to_check = list(check_interval)
            intervals_to_remove = []

            for i in intervals_to_check:
                if ad_test(interval_membership[i]) > self.ad_threshold:
                    to_split = i
                    break
                intervals_to_remove.append(i)

            # remove AFTER iteration
            for i in intervals_to_remove:
                check_interval.discard(i)

            if to_split is None:
                logger.info(f"Convergence after {iteration} iterations.")
                break

            intervals, interval_membership = _split(
                interval_membership, intervals, self.g_overlap, to_split
            )

            check_interval.add(to_split)
            check_interval.add(to_split + 1)

        return intervals


def _split(interval_membership, intervals, g_overlap, index):
    split_interval = _gm_split(
        intervals[index], np.array(interval_membership[index]), g_overlap
    )

    new_intervals = np.delete(intervals, index, axis=0)
    new_intervals = np.insert(new_intervals, index, split_interval, axis=0)

    new_membership = _membership(interval_membership[index], split_interval)
    interval_membership.pop(index)
    interval_membership.insert(index, new_membership[0])
    interval_membership.insert(index + 1, new_membership[1])

    return new_intervals, interval_membership


def _membership(data, intervals):
    """
    Assign data points to intervals.

    Returns
    -------
    list
        List of lists containing data points for each interval
    """
    if hasattr(data, "__iter__"):
        data = np.array(data).flatten()
    else:
        data = np.array([data])

    result = []
    for interval in intervals:
        mask = (data >= interval[0]) & (data <= interval[1])
        members = data[mask]
        result.append(members.tolist())

    return result


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

    Notes
    -----
    When ad_test returns a statistic less than alpha then
    gmapper believes that interval is sufficiently normal and so
    does not split. For small sample sizes we do not bother
    calling the test.

    """
    # convert list to flat numpy array
    data = np.asarray(data).flatten()

    n = len(data)

    if n < 8:
        logger.warning(
            f"Anderson-Darling: Encountered an interval with < 8 data points."
        )
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
