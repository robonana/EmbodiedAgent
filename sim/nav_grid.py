from __future__ import annotations

"""
nav_grid.py — 2-D occupancy-grid path planner + kinematic navigation step.

NavGrid
    Rasterises the ReplicaCAD navmesh onto a regular grid, erodes by the robot
    footprint radius, then runs weighted 8-connected A* with a distance-transform
    cost map.  Visibility-based string-pull removes redundant waypoints.

kinematic_nav_step
    Advances the robot one step via direct qpos updates (bypasses the velocity
    controller, which does not work for XLeRobot in ReplicaCAD).

This is the ManiSkill backend's implementation of BaseToolbox._plan_path (the polyline it
returns is what navigate()'s pure-pursuit loop follows).

The central idea is that there are TWO occupancy grids, and using the right one in the right
place is what makes the planner both safe and able to get through doorways:

    free_before_erosion   walkable floor (dilated navmesh).  A* searches this.
    free                  the above, eroded by the robot radius.  Line-of-sight
                          simplification and the endpoint snap use this.

If A* searched only `free`, a corridor narrower than 2× the robot radius would erode away
entirely and be declared impassable, even though the robot fits. Instead A* may route
through the clearance zone but pays a steep proximity penalty for doing so, so it hugs walls
only when it has no choice. The string-pull step then refuses to shortcut through those
tight regions (it tests against `free`), which keeps the *simplified* path from cutting a
corner into a wall.
"""

import heapq
import math

import numpy as np

from .env import (
    NAV_FWD_M_PER_STEP,
    NAV_ROT_RAD_PER_STEP,
    NAV_STOP_DIST,
    ROT_THRESH,
)


class NavGrid:
    """
    2-D occupancy-grid path planner built from ReplicaCAD navigable positions.

    The navmesh OBJ encodes obstacle-free floor space pre-computed by Habitat
    for the Fetch robot.  We rasterise it onto a regular grid, dilate once to
    fill sub-cell gaps, then erode by the robot footprint radius so the planned
    path stays clear of walls and furniture.

    Weighted A* (8-connected) prefers fully-clear cells but can thread through
    the clearance zone when corridors are tight.  Visibility-based string-pull
    simplification removes redundant waypoints without crossing obstacles.

    Coordinate convention: row = Y axis, col = X axis.
    """

    # Obstacle-proximity penalty weight (metres).  Controls how far the
    # expensive zone extends from walls.  Raise to push paths further from
    # obstacles; lower to allow tighter routing.
    #
    # Enters the cost as 1 + W/(d + cell), so at the wall a step costs ~1 + W/cell (huge),
    # and far from any wall it decays to ~1 (free). This is a soft repulsion, not a hard
    # constraint — which is exactly what lets a doorway remain traversable.
    PROXIMITY_WEIGHT: float = 4.0

    def __init__(self, nav_verts: np.ndarray, robot_radius: float = 0.35):
        """
        Parameters
        ----------
        nav_verts    : (N, 2) float32 array of XY navigable positions (metres).
        robot_radius : inflation radius in metres.  Use ≥ 0.35 m for XLeRobot.
        """
        from scipy.spatial import cKDTree
        from scipy.ndimage import binary_dilation, binary_erosion

        if len(nav_verts) < 4:
            raise ValueError("Too few navigable vertices to build a grid.")

        # ── Choose a cell size that matches the navmesh's own sampling density ──
        # The navmesh is a point cloud with no declared resolution, so infer it: for every
        # vertex take the distance to its nearest neighbour (k=2 because k=1 is itself),
        # then take the 20th percentile. The percentile — rather than the min or the mean —
        # resists both duplicate/near-coincident vertices (which would drive the cell size
        # to ~0) and sparse boundary regions (which would inflate it). Clamped to a sane
        # range so a pathological mesh cannot produce a grid with billions of cells.
        tree = cKDTree(nav_verts)
        nn_dist, _ = tree.query(nav_verts, k=2)
        cell = float(np.percentile(nn_dist[:, 1], 20))
        cell = float(np.clip(cell, 0.05, 0.5))
        self.cell = cell

        # ── Allocate grid ──
        # Pad by the robot radius so erosion has room to work at the map boundary, plus one
        # cell of slack for rounding.
        margin = robot_radius + cell
        self.origin = nav_verts.min(axis=0) - margin
        span = nav_verts.max(axis=0) - nav_verts.min(axis=0) + 2 * margin
        self.W = int(np.ceil(span[0] / cell)) + 2   # cols = X
        self.H = int(np.ceil(span[1] / cell)) + 2   # rows = Y

        # ── Rasterise nav vertices ──
        # Note the index order: free[row, col] == free[y, x]. numpy is row-major and the
        # rest of this class follows that convention; getting it backwards silently
        # transposes the map.
        free = np.zeros((self.H, self.W), dtype=bool)
        cols = np.round((nav_verts[:, 0] - self.origin[0]) / cell).astype(int)
        rows = np.round((nav_verts[:, 1] - self.origin[1]) / cell).astype(int)
        mask = (cols >= 0) & (cols < self.W) & (rows >= 0) & (rows < self.H)
        free[rows[mask], cols[mask]] = True

        # Dilation (1 cell) fills sub-cell gaps between adjacent nav points.
        # Without it, walkable floor comes out speckled — the vertices don't land exactly on
        # cell centres — and A* would have to weave around phantom holes.
        struct = np.ones((3, 3), dtype=bool)   # 8-connected structuring element
        free_dilated = binary_dilation(free, structure=struct, iterations=1)

        # Erosion by robot radius adds obstacle clearance.
        # border_value=False treats everything outside the array as obstacle, so the map
        # edge erodes inward like a wall (rather than the robot being able to plan off-map).
        n_erode = max(1, int(np.round(robot_radius / cell)))
        self.free = binary_erosion(free_dilated, structure=struct,
                                   iterations=n_erode, border_value=False)

        # Keep the un-eroded map: A* searches THIS one (see module docstring).
        self.free_before_erosion = free_dilated
        self.robot_radius = robot_radius

        # Distance transform: continuous distance to nearest obstacle (in cells).
        # Computed on the un-eroded map so the distance is to the *real* wall, which is what
        # the proximity penalty in plan() wants.
        from scipy.ndimage import distance_transform_edt
        self.dist_map = distance_transform_edt(self.free_before_erosion)

        print(f"  NavGrid: {self.W}×{self.H} cells @ {cell:.3f} m/cell"
              f"  free={self.free.sum()}  erode={n_erode} cells")

    # ── visualisation ─────────────────────────────────────────────────────────

    def debug_image(self,
                    waypoints: list[np.ndarray],
                    start_xy: np.ndarray,
                    goal_xy: np.ndarray,
                    target_xy: np.ndarray | None = None,
                    robot_xy: np.ndarray | None = None,
                    save_path: str | None = None) -> np.ndarray:
        """Render the occupancy grid + planned path to an RGB numpy image.

        Diagnostic only — nothing in the planner depends on it. It visualises the *actual*
        A* cost field (not the raw occupancy), which is the quickest way to see why a path
        took an odd-looking detour.
        """
        import matplotlib
        matplotlib.use("Agg")   # headless: no display, render straight to a buffer
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

        # Recompute exactly the per-step cost A* uses, so the picture cannot lie about it.
        dist_m = self.dist_map * self.cell
        cost = 1.0 + self.PROXIMITY_WEIGHT / (dist_m + self.cell)
        cost[~self.free_before_erosion] = np.nan   # NaN renders transparent

        # Cap the colour scale at the cost of a cell exactly one robot-radius from a wall,
        # so the interesting range isn't washed out by the near-infinite cost at the wall.
        vmax = 1.0 + self.PROXIMITY_WEIGHT / (self.robot_radius + self.cell)
        extent = [
            self.origin[0], self.origin[0] + self.W * self.cell,
            self.origin[1], self.origin[1] + self.H * self.cell,
        ]
        # origin="lower" so +Y points up, matching world coordinates rather than image ones.
        im = ax.imshow(cost, origin="lower", extent=extent,
                       interpolation="nearest", cmap="RdYlGn_r",
                       vmin=1.0, vmax=vmax, zorder=1)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="A* step cost")

        # Obstacles as an opaque dark overlay (the cost layer is NaN there).
        obstacle_rgba = np.zeros((self.H, self.W, 4), dtype=np.float32)
        obstacle_rgba[~self.free_before_erosion] = [0.2, 0.2, 0.2, 1.0]
        ax.imshow(obstacle_rgba, origin="lower", extent=extent,
                  interpolation="nearest", zorder=2)

        if len(waypoints) >= 2:
            wps = np.array(waypoints)
            ax.plot(wps[:, 0], wps[:, 1], "b-", linewidth=2, zorder=3,
                    label="path")
            ax.plot(wps[:, 0], wps[:, 1], "b.", markersize=6, zorder=4)

        ax.plot(*start_xy, "go", markersize=12, zorder=5, label="start")
        ax.plot(*goal_xy,  "bs", markersize=12, zorder=5, label="nav goal")
        # The object is drawn separately from the nav goal because they differ: the nav goal
        # is a standoff pose NAV_STOP_DIST away from the object (the dotted circle).
        if target_xy is not None:
            ax.plot(*target_xy, "r*", markersize=16, zorder=5, label="object")
            circle_obj = plt.Circle(target_xy, NAV_STOP_DIST,
                                    color="red", fill=False, linestyle=":",
                                    linewidth=1.0, zorder=4,
                                    label=f"stop={NAV_STOP_DIST:.2f}m")
            ax.add_patch(circle_obj)
        if robot_xy is not None:
            ax.plot(*robot_xy, "c^", markersize=10, zorder=6, label="robot")

        # Robot footprint at the goal — shows at a glance whether the goal actually fits.
        circle = plt.Circle(goal_xy, self.robot_radius,
                             color="blue", fill=False, linestyle="--",
                             linewidth=1.2, zorder=4,
                             label=f"r={self.robot_radius:.2f}m")
        ax.add_patch(circle)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"NavGrid  cell={self.cell:.3f} m  robot_radius={self.robot_radius:.2f} m\n"
            "Green=cheap (far from walls)  Red=expensive (near walls)  "
            "Dark=obstacle"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal")   # metres are metres in both axes
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"  Debug image saved → {save_path}")

        # Rasterise the figure into an RGB array so callers can show it in the viewer or
        # attach it to a log without touching the filesystem.
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w_px, h_px = fig.canvas.get_width_height()
        # .copy() because the buffer is owned by the figure we're about to close.
        img = buf.reshape(h_px, w_px, 4)[..., :3].copy()
        plt.close(fig)   # required, or repeated calls leak figures
        return img

    def log_debug_grid(self, waypoints, start_xy, goal_xy,
                       target_xy=None, robot_xy=None, save_path=None) -> np.ndarray:
        """Render and optionally save the debug grid image."""
        return self.debug_image(waypoints, start_xy, goal_xy,
                                target_xy=target_xy, robot_xy=robot_xy,
                                save_path=save_path)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _w2g(self, xy: np.ndarray) -> tuple[int, int]:
        """World XY → (col, row).  Note the return order is (x, y), not numpy's (row, col)."""
        r = (xy - self.origin) / self.cell
        return int(round(r[0])), int(round(r[1]))

    def _g2w(self, col: int, row: int) -> np.ndarray:
        """(col, row) → world XY (cell centre)."""
        return self.origin + np.array([col, row], dtype=np.float64) * self.cell

    def _snap(self, col: int, row: int,
              grid: np.ndarray | None = None) -> tuple[int, int]:
        """Snap (col, row) to nearest True cell in `grid` (default: self.free).

        Needed because A* endpoints frequently land on obstacle cells: the robot's own pose
        can be inside the eroded zone, and a goal derived from a memory frame may sit
        against a wall. Without snapping, A* would start or end on a blocked cell and
        immediately fail.

        Brute-force nearest search over all free cells. O(free cells) per call, but it runs
        twice per plan (start and goal), so it has never been worth a KD-tree.
        """
        if grid is None:
            grid = self.free
        if 0 <= row < self.H and 0 <= col < self.W and grid[row, col]:
            return col, row   # already valid
        free_rows, free_cols = np.where(grid)
        if len(free_rows) == 0:
            return col, row   # nothing is free; caller will fall back to a direct path
        # Squared distance — no need for the sqrt to find the argmin.
        dists = (free_rows - row) ** 2 + (free_cols - col) ** 2
        idx = int(np.argmin(dists))
        return int(free_cols[idx]), int(free_rows[idx])

    # ── A* ────────────────────────────────────────────────────────────────────

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> list[np.ndarray]:
        """Plan a collision-free path from start to goal.

        Returns a list of XY waypoints (world frame).  Falls back to a
        single-waypoint direct path if A* finds no route.

        Weighted 8-connected A* over `free_before_erosion` with a wall-proximity penalty.
        Because the penalty inflates step costs above the heuristic's scale, the search is
        *inadmissible* — the returned path is not guaranteed shortest. That is intentional:
        we want the path that keeps the robot away from furniture, not the one that shaves
        a few centimetres by scraping along a wall.
        """
        # Snap to the un-eroded map: the robot's current cell is often inside the clearance
        # zone, and snapping it to `free` would teleport the start point metres away.
        sc, sr = self._snap(*self._w2g(start_xy), grid=self.free_before_erosion)
        gc, gr = self._snap(*self._w2g(goal_xy),  grid=self.free_before_erosion)

        if (sc, sr) == (gc, gr):
            return [goal_xy.copy()]   # already there (within one cell)

        SQRT2 = math.sqrt(2.0)
        # Euclidean heuristic in CELL units, matching the base step costs (1 / √2).
        h = lambda c, r: math.hypot(c - gc, r - gr)

        # (f, g, col, row). f first so the heap orders by it; g carried along to detect
        # stale entries below.
        open_set: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (h(sc, sr), 0.0, sc, sr))
        g_score: dict[tuple[int, int], float] = {(sc, sr): 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        while open_set:
            _, g, c, r = heapq.heappop(open_set)

            if (c, r) == (gc, gr):
                # Reconstruct by walking the parent chain back to the start, then reverse.
                path_cells = []
                node = (gc, gr)
                while node in came_from:
                    path_cells.append(node)
                    node = came_from[node]
                path_cells.append((sc, sr))
                path_cells.reverse()
                simplified = self._vis_simplify(path_cells)
                return [self._g2w(c_, r_) for c_, r_ in simplified]

            # Lazy deletion: heapq has no decrease-key, so a node can sit in the heap under
            # an old, worse g. Skip it if we have since found a cheaper route to that cell.
            if g > g_score.get((c, r), float("inf")) + 1e-9:
                continue

            # 8-connected expansion.
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = c + dc, r + dr
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    # Search the UN-eroded map, so narrow doorways stay passable.
                    if not self.free_before_erosion[nr, nc]:
                        continue
                    # Diagonal moves cost √2 — otherwise A* would prefer staircase paths.
                    base = SQRT2 if dc != 0 and dr != 0 else 1.0
                    # Wall-proximity penalty. `+ self.cell` in the denominator keeps the
                    # cost finite for a cell sitting right on the boundary (d = 0).
                    dist_m = float(self.dist_map[nr, nc]) * self.cell
                    step = base * (1.0 + self.PROXIMITY_WEIGHT / (dist_m + self.cell))
                    ng = g + step
                    if ng < g_score.get((nc, nr), float("inf")):
                        g_score[(nc, nr)] = ng
                        came_from[(nc, nr)] = (c, r)
                        heapq.heappush(open_set, (ng + h(nc, nr), ng, nc, nr))

        # Exhausted the open set: goal is unreachable (a closed door, a disconnected room).
        # Hand back the goal as a single waypoint and let the caller's control loop make a
        # best effort — it will stop when it stops making progress.
        print("  [NavGrid] A* found no path, falling back to direct navigation.")
        return [goal_xy.copy()]

    # ── path simplification ───────────────────────────────────────────────────

    def _los_ok(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Bresenham line-of-sight: every cell on the line from a→b must be
        inside self.free (the fully-clear zone, not the clearance zone).

        Tested against the ERODED map, deliberately. A* was allowed to squeeze through the
        clearance zone; a straight-line shortcut must not be. Otherwise the string-pull
        below would happily replace a careful path around a doorframe with a diagonal that
        clips it.

        Standard integer Bresenham: `err` tracks the accumulated error, and the two
        conditional updates step in x and/or y (both, on a diagonal).
        """
        c0, r0 = a
        c1, r1 = b
        dc, dr = abs(c1 - c0), abs(r1 - r0)
        sc = 1 if c0 < c1 else -1
        sr = 1 if r0 < r1 else -1
        err = dc - dr
        c, r = c0, r0
        while True:
            if not (0 <= r < self.H and 0 <= c < self.W):
                return False
            if not self.free[r, c]:
                return False
            if c == c1 and r == r1:
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    def _vis_simplify(self, path_cells: list[tuple[int, int]]
                      ) -> list[tuple[int, int]]:
        """Greedy visibility / string-pull: skip waypoints when LOS holds.

        A* returns one waypoint per grid cell, which is far too dense to drive (hundreds of
        near-collinear points). From each kept waypoint, scan backwards from the END of the
        path for the furthest cell still in line of sight, jump straight there, and repeat.
        The result is a handful of corner points.

        Scanning from the end (rather than forward from i) is what makes it greedy-maximal:
        the first LOS hit found is the furthest reachable one.
        """
        if len(path_cells) <= 2:
            return path_cells
        result = [path_cells[0]]
        i = 0
        while i < len(path_cells) - 1:
            j = len(path_cells) - 1
            while j > i + 1 and not self._los_ok(path_cells[i], path_cells[j]):
                j -= 1
            # j is now the furthest visible cell (worst case j == i+1, the next cell, which
            # is always reachable — so this always advances and cannot loop forever).
            result.append(path_cells[j])
            i = j
        return result


# ── Kinematic navigation step ─────────────────────────────────────────────────

def kinematic_nav_step(agent, bearing: float) -> None:
    """Advance the robot one step toward `bearing` via direct qpos update.

    Uses qpos[0]=root_x, qpos[1]=root_y, qpos[2]=root_z_rotation.
    Kinematic updates bypass the velocity controller, which doesn't work for
    XLeRobot in ReplicaCAD (the scene builder only disables floor-collision
    groups for the Fetch robot, leaving XLeRobot's base stuck against the floor).

    Turn-then-drive, not simultaneous: if the goal is more than ROT_THRESH (~8°) off the
    nose, rotate only; otherwise drive straight only. This is what BaseToolbox._align_yaw
    is working around when it clamps small bearings up to 9° — a bearing below the threshold
    is read here as "aligned, go forward", so a pure rotation must always be requested with
    |bearing| > ROT_THRESH.

    Because it writes qpos directly, the base does not collide with anything. Collision
    avoidance is entirely NavGrid's responsibility — there is no physical fallback.
    """
    qp = agent.robot.get_qpos().cpu().numpy().flatten().copy()
    yaw = qp[2]

    if abs(bearing) > ROT_THRESH:
        # Rotate, capped at the per-step rate so a large bearing doesn't snap round in one
        # frame (which would look wrong and would break any camera capture mid-turn).
        step = np.clip(bearing, -NAV_ROT_RAD_PER_STEP, NAV_ROT_RAD_PER_STEP)
        qp[2] = yaw + step
    else:
        # Aligned: translate along the current heading.
        import math as _math
        qp[0] += NAV_FWD_M_PER_STEP * _math.cos(yaw)
        qp[1] += NAV_FWD_M_PER_STEP * _math.sin(yaw)

    agent.robot.set_qpos(qp)
