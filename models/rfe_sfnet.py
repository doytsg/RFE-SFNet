"""Public RFE-SFNet model entry.

The implementation is kept in ``sds_dsfb_transformer.py`` for backward
compatibility with earlier experiment scripts.
"""

from .sds_dsfb_transformer import RFESFNet, SDSDSFBTransformer

__all__ = ["RFESFNet", "SDSDSFBTransformer"]
