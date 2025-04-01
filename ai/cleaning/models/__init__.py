"""
AI models for medical image cleaning and enhancement.
Includes different model groups for various approaches.
"""

# Import the model groups
from . import novel
from . import foundational
# from . import custom  # Will be implemented in future

__all__ = [
    'novel',
    'foundational',
    # 'custom'
]