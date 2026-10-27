"""Display helpers shared by the command-line report and the dashboard."""

from __future__ import annotations


def money(x: float) -> str:
    """Short money format: 1,234,567 -> '1.23M', 45,600 -> '46k'."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}{x / 1e3:.0f}k"
    return f"{sign}{x:.0f}"


def input_label(path: str) -> str:
    """Readable name for an input path.

    'streams.rooms.demand' -> 'rooms: demand'
    'costs.fixed.staff'    -> 'staff (fixed cost)'
    'streams.rooms.price:multiply[0]' -> 'rooms: price (plan change)'
    """
    base, _, change = path.partition(":")
    parts = base.split(".")
    if parts[0] == "streams" and len(parts) == 3:
        label = f"{parts[1]}: {parts[2].replace('_', ' ')}"
    elif parts[0] == "costs" and len(parts) == 3:
        label = f"{parts[2].replace('_', ' ')} ({parts[1].replace('_', '-')} cost)"
    elif parts[0] == "capacity" and len(parts) == 2:
        label = f"capacity: {parts[1].replace('_', ' ')}"
    else:
        label = base.replace("_", " ")
    return f"{label} (plan change)" if change else label
