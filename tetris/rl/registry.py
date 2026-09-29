"""Which agent class wrote a checkpoint, so the right one can rebuild it.

`DQNAgent.load` already rebuilds *the class it is called on*, which is enough
for anything that subclasses `DQNAgent`. The gap is the loader: every caller
that just wants "the agent in this file" -- the comparison table, the diagnosis,
the viewer -- went through `load_agent_for_evaluation`, which builds a plain
`DQNAgent` whatever the file says. That is right until a campaign trains an arm
whose weights have a different shape: a dueling checkpoint has `trunk` /
`value` / `advantage` instead of `layers`, a feature-plane checkpoint has a
wider input, and `load_state_dict` then fails with a shape error that says
nothing about the cause.

So each agent class registers itself here under the `name` it puts in the
checkpoint's own `agent` field, and the loader looks that name up. A checkpoint
from an agent this code does not know raises with the name in the message,
instead of a mismatch several frames deeper.

Registration happens at import time and the registry is keyed by name only --
last one in wins, so a module reload re-registers rather than duplicating.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional, Type

import torch

#: agent `name` -> the class that builds it.
REGISTRY: Dict[str, Type[Any]] = {}


def register(cls: Type[Any]) -> Type[Any]:
    """Register an agent class under its `name` (used as a decorator)."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{cls!r} has no `name` to register under")
    REGISTRY[name] = cls
    return cls


def agent_class(name: str) -> Type[Any]:
    """The registered class for an agent name, or raise naming what is known."""
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "none"
        raise ValueError(
            f"checkpoint was written by agent {name!r}, which this code does not "
            f"know (known: {known}). Import the module that defines it "
            f"(tetris.rl.variants registers the experiment arms) or load it with "
            f"that class's own `load`."
        )
    return REGISTRY[name]


def checkpoint_agent_name(path) -> Optional[str]:
    """The `agent` field of a checkpoint file, without rebuilding anything."""
    target = pathlib.Path(path)
    if not target.is_file():
        return None
    try:
        payload = torch.load(target, map_location="cpu")
    except Exception:                                  # not a checkpoint at all
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("agent")
    return name if isinstance(name, str) else None


def load_agent(path, device: str = "cpu"):
    """Rebuild the agent a checkpoint holds, whatever class wrote it.

    Reads the file twice -- once for the name, once to build -- which is
    deliberate: the class has to be known *before* the network can be rebuilt,
    and these files are a few megabytes.
    """
    name = checkpoint_agent_name(path)
    if name is None:
        raise ValueError(
            f"{path} is not a checkpoint this code can read (no `agent` field)"
        )
    return agent_class(name).load(path, device=device)


__all__ = [
    "REGISTRY",
    "agent_class",
    "checkpoint_agent_name",
    "load_agent",
    "register",
]
