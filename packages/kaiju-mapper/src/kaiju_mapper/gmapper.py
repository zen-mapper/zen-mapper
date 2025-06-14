import heapq
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import anderson
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

__all__ = ("GMapperCoverScheme", "Interval")


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
    intervals: list[Interval] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.iterations < 1:
            raise ValueError(f"iterations must be > 0, got {self.iterations}")

        if self.max_intervals < 1:
            raise ValueError(f"max_intervals must be > 0, got {self.max_intervals}")

        if self.ad_threshold <= 0:
            raise ValueError(f"ad_threshold must be > 0, got {self.ad_threshold}")

        if self.g_overlap <= 0 or self.g_overlap >= 1:
            raise ValueError(
                f"g_overlap must be in the range (0,1), got {self.g_overlap}"
            )

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
        )

        return [interval.members for interval in self.intervals]


def ad_test(data: np.ndarray) -> float:
    n = len(data)

    if n < 8:
        # The sample size is too small for the ad_test to be meaningful
        return 0

    result = anderson(data)

    return result.statistic * (1 + (4 / n) - (5 / n) ** 2)  # type: ignore


def split(interval: Interval, data: np.ndarray, g_overlap: float) -> list[Interval]:
    masked_data = data[interval.members]
    mean = np.mean(masked_data)
    std = np.std(masked_data, ddof=1)
    m = np.sqrt(2 / np.pi) * std
    c1, c2 = mean + m, mean - m

    gmm = GaussianMixture(
        n_components=2,
        means_init=[[c1], [c2]],
        covariance_type="full",
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
            return []
        except ConvergenceWarning:
            # gmm only found one cluster, refuse to split
            return []

    means: np.ndarray = gmm_result.means_.flatten()  # type: ignore
    covariances: np.ndarray = gmm_result.covariances_.flatten()  # type: ignore

    index = np.argsort(means)
    means = means[index]
    left_mean, right_mean = means[index]
    left_std, right_std = np.sqrt(covariances[index])

    new_upper_bound = left_mean + (1 + g_overlap) * left_std / (
        left_std + right_std
    ) * (right_mean - left_mean)

    new_lower_bound = right_mean - (1 + g_overlap) * right_std / (
        left_std + right_std
    ) * (right_mean - left_mean)

    result = []

    if new_upper_bound < interval.upper_bound:
        result.append(
            make_interval(
                data, interval.lower_bound, new_upper_bound, mask=interval.members
            )
        )

    if new_lower_bound > interval.lower_bound:
        result.append(
            make_interval(
                data, new_lower_bound, interval.upper_bound, mask=interval.members
            )
        )

    return result


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
) -> list[Interval]:
    cover: list[Interval] = []
    heapq.heappush(cover, make_interval(lens, np.min(lens), np.max(lens)))
    iteration = 0
    while True:
        if len(cover) >= max_intervals:
            print("max_intervals hit")
            break

        if iteration > iterations:
            print("max iteration hit")
            break

        if -cover[0].ad_score < ad_threshold:
            print("largest ad_score is below threshold")
            break

        iteration += 1
        interval = heapq.heappop(cover)

        new_intervals = split(interval, lens, g_overlap)

        if len(new_intervals) == 0:
            # splitting the interval resulted in no new intervals, to stop us
            # checking again we put it back on the heap with an impossibly
            # small ad_score
            interval.ad_score = np.inf
            heapq.heappush(cover, interval)

        for interval in new_intervals:
            heapq.heappush(cover, interval)

    return cover
