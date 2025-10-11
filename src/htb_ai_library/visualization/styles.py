"""
Shared styling utilities for Hack The Box themed plots.
"""

from __future__ import annotations

from typing import Optional

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
    "use_htb_style",
]


def use_htb_style() -> None:
    """Apply Hack The Box theme globally to all matplotlib plots.

    This function configures matplotlib's rcParams to use HTB brand colors
    and styling for all subsequent plots. Call this once at the beginning
    of your script to automatically style all figures, axes, and saved images.

    The theme applies:
        - Dark backgrounds (NODE_BLACK) for figures and axes
        - HTB_GREEN titles with bold weight
        - WHITE text for labels and annotations
        - HACKER_GREY for ticks, spines, and grid lines
        - Consistent styling for saved figures

    Examples
    --------
    >>> from htb_ai_library.visualization import use_htb_style
    >>> import matplotlib.pyplot as plt
    >>>
    >>> use_htb_style()  # Apply HTB theme globally
    >>>
    >>> fig, ax = plt.subplots()  # Automatically styled
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> ax.set_title("My Plot")  # Title automatically HTB_GREEN
    >>> plt.savefig("output.png")  # Saves with NODE_BLACK background

    Notes
    -----
    This function modifies matplotlib's global rcParams. To apply styling
    to individual axes (titles, labels, grids), use ``style_htb_axis()``
    in addition to this function.

    See Also
    --------
    style_htb_axis : Apply HTB styling to individual axes with custom labels
    """
    plt.rcParams.update({
        # Figure styling
        'figure.facecolor': NODE_BLACK,
        'figure.edgecolor': NODE_BLACK,

        # Axes styling
        'axes.facecolor': NODE_BLACK,
        'axes.edgecolor': HACKER_GREY,
        'axes.labelcolor': WHITE,
        'axes.titlecolor': HTB_GREEN,
        'axes.titleweight': 'bold',
        'axes.grid': True,

        # Tick styling
        'xtick.color': HACKER_GREY,
        'ytick.color': HACKER_GREY,
        'xtick.labelcolor': HACKER_GREY,
        'ytick.labelcolor': HACKER_GREY,

        # Grid styling
        'grid.color': HACKER_GREY,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',

        # Text styling
        'text.color': WHITE,

        # Legend styling
        'legend.facecolor': NODE_BLACK,
        'legend.edgecolor': HACKER_GREY,
        'legend.framealpha': 0.9,

        # Save figure styling
        'savefig.facecolor': NODE_BLACK,
        'savefig.edgecolor': NODE_BLACK,
        'savefig.bbox': 'tight',
    })
