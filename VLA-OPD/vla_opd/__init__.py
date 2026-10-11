"""Standalone, paper-faithful implementation of VLA-OPD."""

from .losses import VLAOPDLossOutput, sampled_reverse_kl_pg_loss

__all__ = ["VLAOPDLossOutput", "sampled_reverse_kl_pg_loss"]
__version__ = "0.1.0"
