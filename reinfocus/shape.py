"""Defines the possible shapes to render."""

from dataclasses import dataclass

from reinfocus import types as typ

SPHERE = 0
RECTANGLE = 1

@dataclass
class CpuShape:
    """Convenience class for transferring 'polymorphic' geometry to the GPU.
    
    Args:
        parameters: The necessary parameters to bounce a ray off the shape.
        type: The 'polymorphic' type of the shape."""
    parameters: typ.GpuShapeParameters
    type: int
