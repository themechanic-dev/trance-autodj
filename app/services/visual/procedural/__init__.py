"""Procedural visual generators.

Importing this package registers every generator, so
``app.services.visual.procedural.available()`` is complete after one import.
"""

# Imported for their registration side effect; keep the list alphabetical.
from app.services.visual.procedural import domainwarp as _domainwarp  # noqa: F401
from app.services.visual.procedural import flowfield as _flowfield  # noqa: F401
from app.services.visual.procedural import plasma as _plasma  # noqa: F401
from app.services.visual.procedural import reaction_diffusion as _rd  # noqa: F401
from app.services.visual.procedural import tunnel as _tunnel  # noqa: F401
from app.services.visual.procedural import waves as _waves  # noqa: F401
from app.services.visual.procedural.base import (
    ProceduralGenerator,
    RenderContext,
    available,
    get,
    register,
)

__all__ = [
    "ProceduralGenerator",
    "RenderContext",
    "available",
    "get",
    "register",
]
