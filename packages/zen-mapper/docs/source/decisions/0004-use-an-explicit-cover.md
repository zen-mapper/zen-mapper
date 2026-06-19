---
status: "accepted"
date: "2026-06-21"
---
# Use an explicit cover

## Context and Problem Statement

The mapper algorithm needs a cover of the high dimensional data in order to
function. The end user should be able to supply this. There are two orthogonal
questions that need answering.

1. Should the user provide the raw cover or a method for producing a cover?
1. How should the data of a cover be encoded?

This seeks to address the first question.

## Decision Drivers

- Minimal overhead for people implementing new covers
- Flexibility, as many use cases as possible should be enabled
- Type safety, python type checkers should be able to tell what is going on

## Considered Options

- Supply a method for computing the cover
- Supply a realized cover

## Decision Outcome

Chosen option: "Supply a realized cover", because supplying a method for
computing the cover adds no benefits and just complicates the interface.

There was a concern that asking the user to supply a realized cover might
increase the overhead on their side. In practice this seems to be a non-issue.
Regardless the user is going to request a cover from a function, the only
question is what it requires and what it returns. Supplying a cover method
looks like this:

```py
# Supply a method for computing the cover
cover_scheme = Width_Balanced_Cover(num_bins=10, percent_overlap=0.2)
result = mapper( cover = cover_scheme, ...)
```

supplying a realized cover looks like this:

```py
# Supply a cover
cover = width_balanced_cover(projection, num_bins=10, percent_overlap=0.2)
result = mapper(cover = cover, ...)
```

The amount of work from the user is the same in both cases. The only potential
issue would be purely conceptual. The second method requires that you supply
the projection at the time of specifying the cover. We think this is a fine
amount of understanding to require of our users.

Moreover with the current method it is not immediately obvious how one
introspects the computed cover. With the "supply a cover" method this is can be
made much more clear; any function which fits a cover is responsible for
returning the metadata as well. This results in code which looks like the
following:

```py
# Supply a cover
cover, metadata = width_balanced_cover(projection, num_bins=10, percent_overlap=0.2)
result = mapper(cover = cover, ...)
```

Our interface decision here does not put any requirement that there be a
metadata object returned nor makes any requirement of its form. This could lead
to issues down the road if many cover schemes are implemented and they diverge
on this pattern. This is a bridge we can cross if it comes.

## Pros and Cons of the Options

### Supply a method for computing the cover

- Good, because we already have this implemented
- Good, because it lines up with how users conceptualize mapper
- Bad, because it increases the overhead of implementing a new cover
- Bad, because it obscures how to interact with covering metadata

### Supply a realized cover

- Good, because it minimizes overhead of implementing a new cover
- Good, because it simplifies the implementation of the mapper algorithm
- Good, because it makes type hinting easier
- Bad, because it maybe adds conceptual overhead to the user
- Bad, because it maybe does not line up with how users conceptualize mapper

## More Information

[ADR-0003](./0003-allow-user-supplied-covers.md) reasons on allowing user supplied covers
