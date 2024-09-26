---
title: Zen Mapper
---

# Zen Mapper

Zen Mapper is a minimal implementation of the TDA algorithm Mapper. It aims to
be an extensible core which can be used for building out more elaborate mapper
analyses.

> Perfection is achieved, not when there is nothing more to add, but when there
> is nothing left to take away.
>
> Antoine de Saint-Exupéry

```{contents} Table of Contents
:depth: 2
```

## Things left to do

At the moment zen mapper is still in pre-release status. The implementation is
correct (as far as I know) but there are still many things left to do before
it's ready for a "proper" release.

### Design

The api design is not final, this is the number one reason it's not 1.0.0 yet.
I'm not completely sold on the current form of the api.

### Documentation

Write some actual documentation. Right now there is just the auto-generated
docs and those aren't even done! Most functions/classes have docstrings but
some don't. Writing some tutorials and examples should help with the design
point.

### Testing

Polish up the test suite. I'm pretty sure it's testing most of the behavior but
not 100%. I'd say something around 78%.

### Optimization

Preference was given to write the most obvious implementations which would work
for this initial pre-release. This was to make sure everything was working
correctly. There is plenty of room left for some pretty trivial optimizations.

---

```{toctree}
:maxdepth: 2
examples/index
autoapi/index
```
