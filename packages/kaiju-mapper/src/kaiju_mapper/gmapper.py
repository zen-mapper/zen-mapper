import heapq
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import anderson
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from kaiju_mapper.types import Seed

__all__ = ("GMapperCoverScheme", "Interval")


_logger = logging.getLogger("kaiju_mapper")


@dataclass(order=True, slots=True)
class Interval:
    ad_score: float
    members: np.ndarray = field(compare=False)
    lower_bound: float = field(compare=False)
    upper_bound: float = field(compare=False)


@dataclass
class GMapperCoverScheme:
    iterations: int
    max_intervals: int
    ad_threshold: float
    g_overlap: float
    intervals: list[Interval]
    rng: np.random.Generator

    def __init__(
        self,
        iterations: int,
        max_intervals: int,
        ad_threshold: float,
        g_overlap: float,
        seed: Seed | None = None,
    ):
        if iterations < 1:
            raise ValueError(f"iterations must be > 0, got {iterations}")

        if max_intervals < 1:
            raise ValueError(f"max_intervals must be > 0, got {max_intervals}")

        if ad_threshold <= 0:
            raise ValueError(f"ad_threshold must be > 0, got {ad_threshold}")

        if g_overlap <= 0 or g_overlap >= 1:
            raise ValueError(f"g_overlap must be in the range (0,1), got {g_overlap}")

        self.iterations = iterations
        self.max_intervals = max_intervals
        self.ad_threshold = ad_threshold
        self.g_overlap = g_overlap
        self.rng = np.random.default_rng(seed)
        self.intervals = list()

    def __call__(self, data: np.ndarray):
        if data.squeeze().ndim > 1:
            raise ValueError(
                f"Data must be one dimensional, got matrix with shape {data.shape}"
            )
        self.intervals = bfs(
            lens=data.flatten(),
            iterations=self.iterations,
            max_intervals=self.max_intervals,
            ad_threshold=self.ad_threshold,
            g_overlap=self.g_overlap,
            random_state=int(self.rng.integers(0, 4294967295, endpoint=True)),
            # sklearn does not accept the new numpy generators, only ints
        )

        return [interval.members for interval in self.intervals]


def ad_test(data: np.ndarray) -> float:
    n = len(data)

    if n < 8:
        # The sample size is too small for the ad_test to be meaningful
        return 0

    result = anderson(data)

    return result.statistic * (1 + (4 / n) - (5 / n) ** 2)  # type: ignore


def split(
    interval: Interval,
    data: np.ndarray,
    g_overlap: float,
    random_state: int,
) -> tuple[Interval, Interval] | None:
    masked_data = data[interval.members]
    mean = np.mean(masked_data)
    std = np.std(masked_data, ddof=1)
    m = np.sqrt(2 / np.pi) * std
    c1, c2 = mean + m, mean - m

    gmm = GaussianMixture(
        n_components=2,
        means_init=[[c1], [c2]],
        covariance_type="full",
        random_state=random_state,
    )

    with warnings.catch_warnings():
        # This warning is raised when the gmm fails to find two clusters
        # we would like to handle this case so we convert the warning to an
        # exception
        warnings.simplefilter("error", category=ConvergenceWarning)
        try:
            gmm_result = gmm.fit(masked_data.reshape(-1, 1))
        except ValueError:
            # sometimes this method ends up crashing due to intermediate infs and
            # nans. I am not sure what about the input causes this to happen so I
            # don't know how to guard against it
            return None
        except ConvergenceWarning:
            # gmm only found one cluster, refuse to split
            return None

    means: np.ndarray = gmm_result.means_.flatten()  # type: ignore
    covariances: np.ndarray = gmm_result.covariances_.flatten()  # type: ignore

    index = np.argsort(means)
    left_mean, right_mean = means[index]
    left_std, right_std = np.sqrt(covariances[index])

    new_upper_bound = left_mean + (1 + g_overlap) * left_std / (
        left_std + right_std
    ) * (right_mean - left_mean)

    new_lower_bound = right_mean - (1 + g_overlap) * right_std / (
        left_std + right_std
    ) * (right_mean - left_mean)

    if new_upper_bound >= interval.upper_bound:
        return None

    if new_lower_bound <= interval.lower_bound:
        return None

    return make_interval(
        data,
        interval.lower_bound,
        new_upper_bound,
        mask=interval.members,
    ), make_interval(
        data,
        new_lower_bound,
        interval.upper_bound,
        mask=interval.members,
    )


def make_interval(
    data: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    mask: np.ndarray | None = None,
) -> Interval:
    if mask is None:
        mask = np.arange(len(data))

    masked_data = data[mask]
    new_mask = np.flatnonzero(
        (masked_data >= lower_bound) & (masked_data <= upper_bound)
    )
    return Interval(
        ad_score=-ad_test(data[new_mask]),
        members=mask[new_mask],
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def bfs(
    lens: np.ndarray,
    iterations: int,
    max_intervals: int,
    ad_threshold: float,
    g_overlap: float,
    random_state: int,
) -> list[Interval]:
    cover: list[Interval] = []
    heapq.heappush(cover, make_interval(lens, np.min(lens), np.max(lens)))
    iteration = 0
    while True:
        if len(cover) >= max_intervals:
            _logger.info("max_intervals hit")
            break

        if iteration > iterations:
            _logger.info("max iteration hit")
            break

        if -cover[0].ad_score < ad_threshold:
            _logger.info("largest ad_score is below threshold")
            break

        iteration += 1
        interval = heapq.heappop(cover)

        new_intervals = split(interval, lens, g_overlap, random_state=random_state)

        if new_intervals is None:
            # splitting the interval resulted in no new intervals, to stop us
            # checking again we put it back on the heap with an impossibly
            # small ad_score
            interval.ad_score = np.inf
            heapq.heappush(cover, interval)
            continue

        for interval in new_intervals:
            heapq.heappush(cover, interval)

    return cover
