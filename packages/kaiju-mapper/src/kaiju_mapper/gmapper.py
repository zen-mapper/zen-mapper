import heapq
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from scipy.stats import anderson
from sklearn.mixture import GaussianMixture


def ad_test(data: np.ndarray) -> float:
    n = len(data)

    if n == 0:
        return 0

    result = anderson(data)

    return result.statistic * (2 + 4 / n - 25 / (n**2))  # type: ignore


@dataclass(order=True, slots=True)
class Interval:
    ad_score: float
    members: np.ndarray = field(compare=False)
    lower_bound: float = field(compare=False)
    upper_bound: float = field(compare=False)

    def split(self, data: np.ndarray, g_overlap: float) -> list[Self]:
        masked_data = data[self.members]
        mean = np.mean(masked_data)
        std = np.std(masked_data, ddof=1)
        m = np.sqrt(2 / np.pi) * std
        c1, c2 = mean + m, mean - m
        gmm = GaussianMixture(
            n_components=2,
            means_init=[[c1], [c2]],
            covariance_type="full",
        ).fit(masked_data.reshape(-1, 1))

        means: np.ndarray = gmm.means_.flatten()  # type: ignore
        covariances: np.ndarray = gmm.covariances_.flatten()  # type: ignore

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

        if new_upper_bound < self.upper_bound:
            result.append(
                make_interval(
                    data, self.lower_bound, new_upper_bound, members=self.members
                )
            )

        if new_lower_bound > self.lower_bound:
            result.append(
                make_interval(
                    data, new_lower_bound, self.upper_bound, members=self.members
                )
            )

        return result


def make_interval(
    data: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    members: np.ndarray | None = None,
) -> Interval:
    if members is None:
        members = np.arange(len(data))

    masked_data = data[members]
    mask = (masked_data >= lower_bound) & (masked_data <= upper_bound)
    members = members[mask]
    return Interval(
        ad_score=-ad_test(data[members]),
        members=members,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def bfs(
    lens: np.ndarray,
    iterations: int = 10,
    max_intervals: int = 20,
    ad_threshold: float = 10,
    g_overlap=0.1,
):
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
        new_intervals = interval.split(lens, g_overlap)

        print(f"{len(new_intervals)=}")

        if len(new_intervals) == 0:
            # splitting the interval resulted in no new intervals, to stop us
            # checking again we put it back on the heap with an impossibly
            # large ad_score
            interval.ad_score = np.inf
            heapq.heappush(cover, interval)

        for interval in new_intervals:
            heapq.heappush(cover, interval)

    return [interval.members for interval in cover]
