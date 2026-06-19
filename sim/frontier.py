from __future__ import annotations

"""
frontier.py — 2-D occupancy mapping + frontier detection for active exploration.

Builds a top-down (world XZ) occupancy grid incrementally from the robot's
head-depth observations, using the same Habitat camera convention as
HabitatToolbox._estimate_object_xy_from_depth_hab:

    forward = -Z_cam,   depth d → cam point (xs·d·f_inv, ys·d·f_inv, -d)
    world   = T_cw · cam_point          (T_cw = camera-to-world)

Cells are UNKNOWN until observed, then FREE (seen empty floor / line-of-sight)
or OCC (a depth ray terminated on a vertical surface). A *frontier* is a FREE
cell touching UNKNOWN space — the boundary of the explored region. Sampling
navigable viewpoints near frontiers and scoring them by

    score = information_gain − λ · travel_cost

drives the robot toward the most informative unexplored area (the classic
Yamauchi frontier-exploration objective).
"""

import math
import numpy as np

UNKNOWN, FREE, OCC = -1, 0, 1


def _bresenham(x0: int, z0: int, x1: int, z1: int):
    """Integer grid cells along the line (x0,z0)->(x1,z1), endpoints included."""
    cells = []
    dx = abs(x1 - x0); dz = abs(z1 - z0)
    sx = 1 if x0 < x1 else -1
    sz = 1 if z0 < z1 else -1
    err = dx - dz
    x, z = x0, z0
    while True:
        cells.append((x, z))
        if x == x1 and z == z1:
            break
        e2 = 2 * err
        if e2 > -dz:
            err -= dz; x += sx
        if e2 < dx:
            err += dx; z += sz
    return cells


class OccupancyMap:
    """Top-down XZ occupancy grid. Indexed [ix, iz]; world Y is vertical."""

    def __init__(self, xmin, zmin, xmax, zmax, res: float = 0.10):
        self.res = float(res)
        self.xmin = float(xmin)
        self.zmin = float(zmin)
        self.nx = max(1, int(math.ceil((float(xmax) - self.xmin) / self.res)))
        self.nz = max(1, int(math.ceil((float(zmax) - self.zmin) / self.res)))
        self.grid = np.full((self.nx, self.nz), UNKNOWN, dtype=np.int8)

    # ── coordinate transforms ────────────────────────────────────────────────
    def w2g(self, x, z):
        return (int((x - self.xmin) / self.res), int((z - self.zmin) / self.res))

    def g2w(self, ix, iz):
        return (self.xmin + (ix + 0.5) * self.res, self.zmin + (iz + 0.5) * self.res)

    def in_bounds(self, ix, iz):
        return 0 <= ix < self.nx and 0 <= iz < self.nz

    # ── sensor integration ───────────────────────────────────────────────────
    def integrate(self, depth, rot, trans, floor_y,
                  max_range: float = 4.0, hfov: float = math.pi / 2.0,
                  stride: int = 4, obstacle_lo: float = 0.12,
                  obstacle_hi: float = 1.8):
        """Fuse one depth frame. rot (3x3) / trans (3,) are the camera-to-world
        rotation and translation (Habitat depth_rot / depth_trans)."""
        H, W = depth.shape
        f_inv = math.tan(hfov / 2.0)
        cam = np.asarray(trans, dtype=np.float64).flatten()
        cix, ciz = self.w2g(cam[0], cam[2])

        us = np.arange(0, W, stride)
        vs = np.arange(0, H, stride)
        uu, vv = np.meshgrid(us, vs)
        d = depth[vv, uu].astype(np.float64)
        valid = (d > 0.05) & (d < max_range)

        # Mark the camera cell free even if nothing is seen.
        if self.in_bounds(cix, ciz) and self.grid[cix, ciz] != OCC:
            self.grid[cix, ciz] = FREE
        if not valid.any():
            return

        uu, vv, d = uu[valid], vv[valid], d[valid]
        xs_n = 2.0 * uu / (W - 1) - 1.0
        ys_n = 1.0 - 2.0 * vv / (H - 1)
        cam_pts = np.stack([xs_n * d * f_inv, ys_n * d * f_inv, -d,
                            np.ones_like(d)], axis=0)        # 4 x N
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(rot, dtype=np.float64)
        T[:3, 3] = cam
        pw = T @ cam_pts                                     # 4 x N
        wx, wy, wz = pw[0], pw[1], pw[2]
        is_obstacle = (wy > floor_y + obstacle_lo) & (wy < floor_y + obstacle_hi)

        for i in range(wx.shape[0]):
            tix, tiz = self.w2g(wx[i], wz[i])
            cells = _bresenham(cix, ciz, tix, tiz)
            for (ix, iz) in cells[:-1]:                      # line of sight = free
                if self.in_bounds(ix, iz) and self.grid[ix, iz] != OCC:
                    self.grid[ix, iz] = FREE
            ex, ez = cells[-1]
            if self.in_bounds(ex, ez):
                if is_obstacle[i]:
                    self.grid[ex, ez] = OCC            # a wall/obstacle hit
                elif self.grid[ex, ez] != OCC:
                    self.grid[ex, ez] = FREE           # floor hit — but never erase a wall

    # ── frontier queries ─────────────────────────────────────────────────────
    def frontiers(self) -> np.ndarray:
        """FREE cells adjacent (4-connected) to UNKNOWN. Returns (M,2) of (ix,iz)."""
        free = self.grid == FREE
        unk = self.grid == UNKNOWN
        fr = np.zeros_like(free)
        fr[:-1, :] |= free[:-1, :] & unk[1:, :]
        fr[1:, :] |= free[1:, :] & unk[:-1, :]
        fr[:, :-1] |= free[:, :-1] & unk[:, 1:]
        fr[:, 1:] |= free[:, 1:] & unk[:, :-1]
        return np.argwhere(fr)

    def frontier_clusters(self, min_size: int = 3):
        """Group frontier cells into 8-connected clusters. Returns a list of
        (world_x, world_z, size) per cluster, largest first, where (world_x,
        world_z) is the cluster *medoid* — an actual frontier cell on the
        boundary, NOT the centroid (a ring-shaped frontier has its centroid in
        already-explored interior, which would send the robot nowhere new).
        Cluster size (boundary length) ∝ how much UNKNOWN lies behind it."""
        fr = self.frontiers()
        if len(fr) == 0:
            return []
        fset = {(int(i), int(j)) for i, j in fr}
        seen = set()
        clusters = []
        for cell in fset:
            if cell in seen:
                continue
            stack = [cell]; seen.add(cell); comp = []
            while stack:
                cx, cz = stack.pop()
                comp.append((cx, cz))
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nb = (cx + dx, cz + dz)
                        if nb in fset and nb not in seen:
                            seen.add(nb); stack.append(nb)
            if len(comp) >= min_size:
                arr = np.array(comp, dtype=float)
                ctr = arr.mean(axis=0)
                # medoid: the real frontier cell closest to the centroid
                medoid = arr[np.argmin(((arr - ctr) ** 2).sum(axis=1))]
                wx, wz = self.g2w(medoid[0], medoid[1])
                clusters.append((wx, wz, len(comp)))
        clusters.sort(key=lambda c: c[2], reverse=True)
        return clusters

    def count_unknown_near(self, x, z, radius: float) -> int:
        """Approx information gain: UNKNOWN cells within `radius` m of (x,z)."""
        ix, iz = self.w2g(x, z)
        r = max(1, int(radius / self.res))
        sub = self.grid[max(0, ix - r):ix + r + 1, max(0, iz - r):iz + r + 1]
        return int(np.count_nonzero(sub == UNKNOWN))

    def to_rgb(self, visited=None, marks=None, robot=None) -> np.ndarray:
        """Render the map to an oriented RGB array: UNKNOWN=grey, FREE=white,
        OCC=black; visited viewpoints red, `marks` (x,z,(r,g,b)) colored dots,
        `robot` (x,z) a yellow dot. X horizontal, Z vertical (image convention)."""
        img = np.full((self.nx, self.nz, 3), 128, dtype=np.uint8)   # unknown
        img[self.grid == FREE] = (255, 255, 255)
        img[self.grid == OCC] = (0, 0, 0)

        def stamp(x, z, color, r=1):
            ix, iz = self.w2g(x, z)
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    if self.in_bounds(ix + a, iz + b):
                        img[ix + a, iz + b] = color
        for (vx, vz) in (visited or []):
            stamp(vx, vz, (255, 0, 0), 1)
        for (mx, mz, c) in (marks or []):
            stamp(mx, mz, c, 2)
        if robot is not None:
            stamp(robot[0], robot[1], (255, 220, 0), 2)
        return np.transpose(img, (1, 0, 2))[::-1]

    def save_png(self, path, visited=None, marks=None):
        """Dump the map to a PNG (see to_rgb)."""
        import imageio.v2 as imageio
        imageio.imwrite(str(path), self.to_rgb(visited=visited, marks=marks))

    def stats(self) -> dict:
        return {"free": int(np.count_nonzero(self.grid == FREE)),
                "occ": int(np.count_nonzero(self.grid == OCC)),
                "unknown": int(np.count_nonzero(self.grid == UNKNOWN))}
