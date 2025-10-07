"""
Color palette utilities for Hack The Box themed visualizations.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

HTB_GREEN = "#9fef00"
NODE_BLACK = "#141d2b"
HACKER_GREY = "#a4b1cd"
WHITE = "#ffffff"
AZURE = "#0086ff"
NUGGET_YELLOW = "#ffaf00"
MALWARE_RED = "#ff3e3e"
VIVID_PURPLE = "#9f00ff"
AQUAMARINE = "#2ee7b6"

_COLOR_PALETTE: dict[str, str] = {
    "HTB_GREEN": HTB_GREEN,
    "NODE_BLACK": NODE_BLACK,
    "HACKER_GREY": HACKER_GREY,
    "WHITE": WHITE,
    "AZURE": AZURE,
    "NUGGET_YELLOW": NUGGET_YELLOW,
    "MALWARE_RED": MALWARE_RED,
    "VIVID_PURPLE": VIVID_PURPLE,
    "AQUAMARINE": AQUAMARINE,
}

HTB_PALETTE: Mapping[str, str] = MappingProxyType(_COLOR_PALETTE)


def get_color(name: str) -> str:
    """
    Retrieve a Hack The Box brand color by name.

    Parameters
    ----------
    name : str
        Case-insensitive color identifier such as ``"HTB_GREEN"``.

    Returns
    -------
    str
        Hexadecimal color string including the leading hash.

    Raises
    ------
    KeyError
        If the requested color name does not exist in the palette.

    Examples
    --------
    >>> get_color("htb_green")
    '#9fef00'
    """
    normalized = name.upper()
    if normalized not in HTB_PALETTE:
        raise KeyError(f"Unknown color '{name}'. Available keys: {sorted(HTB_PALETTE)}")
    return HTB_PALETTE[normalized]


def get_color_palette() -> Mapping[str, str]:
    """
    Provide a read-only mapping of Hack The Box color names to hex strings.

    Returns
    -------
    Mapping[str, str]
        Read-only view keyed by uppercase color names.

    Examples
    --------
    >>> palette = get_color_palette()
    >>> palette["HTB_GREEN"]
    '#9fef00'
    """
    return HTB_PALETTE
