from __future__ import annotations

import logging

from .adapters import sk_learn, to_networkx
from .cover import (
    data_balanced_cover,
    rectangular_cover,
    width_balanced_cover,
)
from .mapper import mapper
from .types import MapperResult

__all__ = [
    "MapperResult",
    "data_balanced_cover",
    "width_balanced_cover",
    "mapper",
    "rectangular_cover",
    "sk_learn",
    "to_networkx",
]

__version__ = "0.3.0"

logger = logging.getLogger("zen_mapper")
logger.addHandler(logging.NullHandler())
