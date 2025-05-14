# Clusters return both clusters and metadata

## Context and Problem Statement

It is sometimes necessary to observe information from the clustering stage of
the mapper pipeline. For example:
- centroid information, useful for visualization
- dendrograms, useful for parameter selection
- core distances, useful for filtering

As zen mapper aims to be a flexible core for building mapper pipelines we need
to allow for authors to somehow pass this information out of the pipeline for
further inspection.

## Decision Drivers

- Minimal overhead for people implementing new clustering algorithms
- Flexibility to allow any type of metadata to be returned
- Type safety, python should be able to deduce what type of metadata is being
  used

## Considered Options

- Custom `ClusterResult` class / interface
- Return a both clusters and a metadata object

## Decision Outcome

Return a both clusters and a metadata object

- no need to define a new class just to wrap your data, just return a tuple.
  This minimizes overhead for cluster algorithm authors
- no restrictions on the type of metadata returned
- python generics are able to infer the type of metadata being used
