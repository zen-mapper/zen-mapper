# Clustering algorithms receive global dataset

## Context and Problem Statement

Sometimes the data coming into a mapper pipeline is not point cloud data. The
most common example of this would be a distance matrix. As zen mapper aims to
be a flexible core for building mapper pipelines we need to allow for these
types of inputs.

As currently written the clustering protocol does not facilitate this. The
`mapper` method does the indexing into the high dimensional data which requires
a fixed understanding of what that high dimensional data represents.

## Decision Drivers

- Minimal overhead for people implementing new clustering algorithms
- Flexibility, as many use cases as possible should be enabled
- Type safety, python type checkers should be able to tell what is going on

## Considered Options

- Change the cluster protocol to receive global dataset
- Implement multiple clustering protocols for different cases

## Decision Outcome

Change the cluster protocol to receive global dataset

By changing the cluster protocol to take the entire high dimensional dataset
along with the indices of the desired subset to cluster it is up to the cluster
author what the high dimensional data represents.

## Consequences

- Good, because it keeps the core algorithm simple
- Good, because it allows for many use cases
- Bad, because the type hinting is more complicated
- Bad, because potentially cryptic error messages
