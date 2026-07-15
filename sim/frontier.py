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

Coordinate note, and the single easiest thing to get wrong here: Habitat is Y-up, so the
FLOOR PLANE is XZ, not XY. Every grid index in this file is (ix, iz) and world Y is only
ever used to decide whether a returned point is an obstacle (at torso height) or floor.
That is why this module says "XZ" everywhere the rest of the codebase says "XY".
"""

import math
import numpy as np

# Cell states. UNKNOWN is -1 rather than 0 so that `grid == FREE` and a zero-initialised
# array can never be confused; the map starts entirely unknown.
UNKNOWN, FREE, OCC = -1, 0, 1


def _bresenham(x0: int, z0: int, x1: int, z1: int):
    """Integer grid cells along the line (x0,z0)->(x1,z1), endpoints included.

    Used for ray-casting: every cell a depth ray passes THROUGH is empty space (the ray
    would have stopped otherwise), and the final cell is where it hit something. Integer-only
    arithmetic, so no floating-point drift can make a ray skip a cell.
    """
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
        """Bounds are in world metres; res is metres per cell (10 cm default — fine enough
        to resolve a doorway, coarse enough that a whole apartment is a few hundred cells)."""
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
        # +0.5 → the CENTRE of the cell, not its corner. Matters when a grid cell becomes a
        # navigation goal: the corner can lie on the far side of a wall.
        return (self.xmin + (ix + 0.5) * self.res, self.zmin + (iz + 0.5) * self.res)

    def in_bounds(self, ix, iz):
        return 0 <= ix < self.nx and 0 <= iz < self.nz

    # ── sensor integration ───────────────────────────────────────────────────
    def integrate(self, depth, rot, trans, floor_y,
                  max_range: float = 4.0, hfov: float = math.pi / 2.0,
                  stride: int = 4, obstacle_lo: float = 0.12,
                  obstacle_hi: float = 1.8):
        """Fuse one depth frame. rot (3x3) / trans (3,) are the camera-to-world
        rotation and translation (Habitat depth_rot / depth_trans).

        Ray-casting fusion: un-project every (strided) depth pixel to a world point, then
        mark every cell along the ray FREE and the terminal cell FREE-or-OCC depending on
        the point's HEIGHT. Height is the whole trick — a ray ending on the floor tells us
        that patch of floor is walkable, while a ray ending at torso height tells us there
        is an obstacle there. Without the height test, the floor itself would be mapped as
        one giant wall.

        stride=4 subsamples the depth image 16× (4× in each axis). A 4-cell-per-metre grid
        cannot use full resolution anyway, and this keeps the per-frame cost small enough to
        run inside the control loop.
        """
        H, W = depth.shape
        # Habitat's projection: with a square sensor and hfov, the normalised image plane
        # spans ±tan(hfov/2) at unit depth. f_inv is that half-extent.
        f_inv = math.tan(hfov / 2.0)
        cam = np.asarray(trans, dtype=np.float64).flatten()
        cix, ciz = self.w2g(cam[0], cam[2])   # ray origin, in grid cells

        # Subsample the depth image on a regular lattice.
        us = np.arange(0, W, stride)
        vs = np.arange(0, H, stride)
        uu, vv = np.meshgrid(us, vs)
        d = depth[vv, uu].astype(np.float64)
        # Reject sensor invalids (0 = no return) and anything beyond max_range, where depth
        # is too noisy to trust and the ray would smear FREE across unobserved space.
        valid = (d > 0.05) & (d < max_range)

        # Mark the camera cell free even if nothing is seen.
        # The robot is standing there, so it is walkable by definition — this seeds the map
        # on the very first frame and guarantees at least one FREE cell exists.
        if self.in_bounds(cix, ciz) and self.grid[cix, ciz] != OCC:
            self.grid[cix, ciz] = FREE
        if not valid.any():
            return

        uu, vv, d = uu[valid], vv[valid], d[valid]
        # Pixel → normalised device coords in [-1, 1]. Note ys is FLIPPED (1 - 2v/…):
        # image rows increase downward, world/camera Y increases upward.
        xs_n = 2.0 * uu / (W - 1) - 1.0
        ys_n = 1.0 - 2.0 * vv / (H - 1)
        # Camera-frame points. -d on the Z axis because Habitat's camera looks down -Z.
        # Homogeneous (the trailing ones row) so the 4×4 transform applies directly.
        cam_pts = np.stack([xs_n * d * f_inv, ys_n * d * f_inv, -d,
                            np.ones_like(d)], axis=0)        # 4 x N
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(rot, dtype=np.float64)
        T[:3, 3] = cam
        pw = T @ cam_pts                                     # 4 x N
        wx, wy, wz = pw[0], pw[1], pw[2]
        # The height test. Below obstacle_lo (12 cm) is floor/skirting — walkable. Above
        # obstacle_hi (1.8 m) is ceiling/high shelves — the robot passes under them, so they
        # are not obstacles either. Only the band between the two blocks the base.
        is_obstacle = (wy > floor_y + obstacle_lo) & (wy < floor_y + obstacle_hi)

        for i in range(wx.shape[0]):
            tix, tiz = self.w2g(wx[i], wz[i])
            cells = _bresenham(cix, ciz, tix, tiz)
            for (ix, iz) in cells[:-1]:                      # line of sight = free
                # `!= OCC` — never downgrade a wall. One stray ray grazing past a wall edge
                # would otherwise punch a hole through it and invite the planner to path
                # straight through.
                if self.in_bounds(ix, iz) and self.grid[ix, iz] != OCC:
                    self.grid[ix, iz] = FREE
            ex, ez = cells[-1]                               # the cell the ray terminated in
            if self.in_bounds(ex, ez):
                if is_obstacle[i]:
                    self.grid[ex, ez] = OCC            # a wall/obstacle hit
                elif self.grid[ex, ez] != OCC:
                    self.grid[ex, ez] = FREE           # floor hit — but never erase a wall

    # ── frontier queries ─────────────────────────────────────────────────────
    def frontiers(self) -> np.ndarray:
        """FREE cells adjacent (4-connected) to UNKNOWN. Returns (M,2) of (ix,iz).

        The frontier is the boundary of what we know: standing on a FREE cell that touches
        UNKNOWN space and looking outward is, by construction, how you see something new.

        Implemented with four shifted array comparisons rather than a loop — each line ORs
        in "this FREE cell has an UNKNOWN neighbour in direction d" for one of the four
        directions. Note a FREE cell beside an OCC cell is NOT a frontier: a wall is
        explored, not unknown.
        """
        free = self.grid == FREE
        unk = self.grid == UNKNOWN
        fr = np.zeros_like(free)
        fr[:-1, :] |= free[:-1, :] & unk[1:, :]    # unknown to the +x
        fr[1:, :] |= free[1:, :] & unk[:-1, :]     # unknown to the -x
        fr[:, :-1] |= free[:, :-1] & unk[:, 1:]    # unknown to the +z
        fr[:, 1:] |= free[:, 1:] & unk[:, :-1]     # unknown to the -z
        return np.argwhere(fr)

    def frontier_clusters(self, min_size: int = 3):
        """Group frontier cells into 8-connected clusters. Returns a list of
        (world_x, world_z, size) per cluster, largest first, where (world_x,
        world_z) is the cluster *medoid* — an actual frontier cell on the
        boundary, NOT the centroid (a ring-shaped frontier has its centroid in
        already-explored interior, which would send the robot nowhere new).
        Cluster size (boundary length) ∝ how much UNKNOWN lies behind it.

        min_size filters out speckle: isolated one- or two-cell frontiers are almost always
        depth noise or a sliver behind a chair leg, not a room worth crossing the apartment
        for.

        The grouping is an iterative flood fill (explicit stack, not recursion — a large
        frontier would blow the Python stack).
        """
        fr = self.frontiers()
        if len(fr) == 0:
            return []
        fset = {(int(i), int(j)) for i, j in fr}   # set → O(1) membership in the fill
        seen = set()
        clusters = []
        for cell in fset:
            if cell in seen:
                continue
            # Flood-fill one 8-connected component.
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
                # (see the docstring — the centroid itself may not be a frontier at all).
                medoid = arr[np.argmin(((arr - ctr) ** 2).sum(axis=1))]
                wx, wz = self.g2w(medoid[0], medoid[1])
                clusters.append((wx, wz, len(comp)))
        # Largest first: a long frontier borders a lot of unexplored space.
        clusters.sort(key=lambda c: c[2], reverse=True)
        return clusters

    def count_unknown_near(self, x, z, radius: float) -> int:
        """Approx information gain: UNKNOWN cells within `radius` m of (x,z).

        The "information gain" term of the exploration objective. Approximate on two counts:
        it counts a square window rather than a disc, and it ignores visibility (some of
        those cells are behind walls and would not actually be revealed). Both are fine —
        this only has to *rank* candidate viewpoints, not predict the true gain.

        max(0, …) on the low index guards the window at the map edge; the high index is
        allowed to overshoot because numpy clamps slice ends silently.
        """
        ix, iz = self.w2g(x, z)
        r = max(1, int(radius / self.res))
        sub = self.grid[max(0, ix - r):ix + r + 1, max(0, iz - r):iz + r + 1]
        return int(np.count_nonzero(sub == UNKNOWN))

    def to_rgb(self, visited=None, marks=None, robot=None) -> np.ndarray:
        """Render the map to an oriented RGB array: UNKNOWN=grey, FREE=white,
        OCC=black; visited viewpoints red, `marks` (x,z,(r,g,b)) colored dots,
        `robot` (x,z) a yellow dot. X horizontal, Z vertical (image convention).
        """
        img = np.full((self.nx, self.nz, 3), 128, dtype=np.uint8)   # unknown
        img[self.grid == FREE] = (255, 255, 255)
        img[self.grid == OCC] = (0, 0, 0)

        def stamp(x, z, color, r=1):
            """Paint a (2r+1)² block so a single cell is actually visible at map scale."""
            ix, iz = self.w2g(x, z)
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    if self.in_bounds(ix + a, iz + b):
                        img[ix + a, iz + b] = color
        # Draw order = priority: robot on top of marks on top of visited.
        for (vx, vz) in (visited or []):
            stamp(vx, vz, (255, 0, 0), 1)
        for (mx, mz, c) in (marks or []):
            stamp(mx, mz, c, 2)
        if robot is not None:
            stamp(robot[0], robot[1], (255, 220, 0), 2)
        # The array is indexed [ix, iz] but an image is [row, col] = [z, x], so transpose;
        # then [::-1] flips vertically so +Z points up rather than down the screen.
        return np.transpose(img, (1, 0, 2))[::-1]

    def save_png(self, path, visited=None, marks=None):
        """Dump the map to a PNG (see to_rgb)."""
        import imageio.v2 as imageio
        imageio.imwrite(str(path), self.to_rgb(visited=visited, marks=marks))

    def stats(self) -> dict:
        """Coverage counters. The unknown count trending to zero means exploration is done."""
        return {"free": int(np.count_nonzero(self.grid == FREE)),
                "occ": int(np.count_nonzero(self.grid == OCC)),
                "unknown": int(np.count_nonzero(self.grid == UNKNOWN))}
