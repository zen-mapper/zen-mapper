---
status: "accepted"
date: "2026-06-21"
---
# Use numpy array to encode cover elements

## Context and Problem Statement

The mapper algorithm needs a cover of the high dimensional data in order to
function. We need a way to represent cover elements.

## Decision Drivers

- Minimal overhead for people implementing new covers

## Considered Options

- Use a numpy array with indices into the original data

## Decision Outcome

Chosen option: "Use a numpy array with indices into the original data", because
this is the most obvious solution and it is unclear the rest of the mapper
pipeline scales enough for this to be the bottleneck.

### Consequences

- Good, because easy to implement.
- Good, because easy to work with.
- Bad, because invalid states are representable (`[1, 1, 2]` is not a set).
- Bad, because it requires the data to fit in memory. If this is a concern it
  is unclear that zen mapper is the right tool for the job.

## More Information

- [ADR-0003](./0003-allow-user-supplied-covers.md) reasons on allowing user supplied covers
- [ADR-0004](./0004-use-an-explicit-cover.md) reasons to supply a cover instead of a scheme
