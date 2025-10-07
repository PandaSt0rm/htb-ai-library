"""
Shared styling utilities for Hack The Box themed plots.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..utils.palette import (
    HTB_GREEN,
    NODE_BLACK,
    HACKER_GREY,
    WHITE,
    AZURE,
    NUGGET_YELLOW,
    MALWARE_RED,
    VIVID_PURPLE,
    AQUAMARINE,
)

__all__ = [
    "HTB_GREEN",
    "NODE_BLACK",
    "HACKER_GREY",
    "WHITE",
    "AZURE",
    "NUGGET_YELLOW",
    "MALWARE_RED",
    "VIVID_PURPLE",
    "AQUAMARINE",
    "apply_htb_axes_style",
]


def apply_htb_axes_style(ax: plt.Axes) -> None:
    """
    Apply the Hack The Box visual identity to a single axes instance.
    """
    ax.set_facecolor(NODE_BLACK)
    ax.tick_params(colors=HACKER_GREY)
    for spine in ax.spines.values():
        spine.set_color(HACKER_GREY)
    ax.grid(True, color=HACKER_GREY, linestyle="--", alpha=0.25)
