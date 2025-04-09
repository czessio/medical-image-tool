"""
AI models for medical image cleaning and enhancement.
Includes different model groups for various approaches.
"""

# Import the model groups
from . import foundational
try:
    from . import novel
except ImportError:
    # Novel models may not be available
    pass

__all__ = [
    'foundational',
    'novel'
]