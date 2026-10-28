"""DewarpNet (ICCV 2019): the document's 3D surface, then a backward map.

Vendored from https://github.com/cvlab-stonybrook/DewarpNet (MIT). The shape
network predicts a 3D coordinate map of the sheet; the texture-mapping network
turns that into the sampling grid that flattens it. Weights:
`unetnc_doc3d.pkl` and `dnetccnl_doc3d.pkl`.
"""
from .densenetccnl import dnetccnl
from .unetnc import UnetGenerator

__all__ = ["dnetccnl", "UnetGenerator"]
