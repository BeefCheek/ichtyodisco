"""Group / shoaling metrics computed on 3D positions in tank units."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class GroupMetrics:
    count: int = 0
    centroid: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cohesion: float = 0.0  # mean distance to centroid
    nnd: float = 0.0  # mean nearest-neighbour distance
    polarization: float = 0.0  # 0 = random headings, 1 = all swimming the same way
    activity: float = 0.0  # mean speed in body lengths / s
    spread_y: float = 0.0  # vertical spread (std of y)
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (a, b, dist) close pairs


def group_metrics(
    ids: list[int],
    positions: np.ndarray,  # N x 3
    velocities: np.ndarray,  # N x 3
    speeds_bls: np.ndarray,  # N
    close_distance: float = 0.1,
) -> GroupMetrics:
    n = len(ids)
    g = GroupMetrics(count=n)
    if n == 0:
        return g
    g.activity = float(np.mean(speeds_bls))
    c = positions.mean(axis=0)
    g.centroid = (float(c[0]), float(c[1]), float(c[2]))
    g.cohesion = float(np.linalg.norm(positions - c, axis=1).mean())
    g.spread_y = float(positions[:, 1].std())
    norms = np.linalg.norm(velocities, axis=1)
    moving = norms > 1e-3
    if moving.sum() >= 2:
        units = velocities[moving] / norms[moving, None]
        g.polarization = float(np.linalg.norm(units.mean(axis=0)))
    if n >= 2:
        d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        g.nnd = float(d.min(axis=1).mean())
        ii, jj = np.where(np.triu(d < close_distance, k=1))
        g.pairs = [(ids[i], ids[j], float(d[i, j])) for i, j in zip(ii, jj, strict=True)]
    return g
