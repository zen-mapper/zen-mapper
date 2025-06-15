import numpy as np

type Seed = int | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator
"""Seeds for random generator

Technically this is overly restrictive at the moment. I could not figure out
how to type hint `np.array_like[int]`. I doubt anyone will notice.
"""
