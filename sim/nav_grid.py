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

        # Infer grid cell size from nearest-neighbour distances (20th percentile)
        tree = cKDTree(nav_verts)
        nn_dist, _ = tree.query(nav_verts, k=2)
        cell = float(np.percentile(nn_dist[:, 1], 20))
        cell = float(np.clip(cell, 0.05, 0.5))
        self.cell = cell

        # Allocate grid
        margin = robot_radius + cell
        self.origin = nav_verts.min(axis=0) - margin
        span = nav_verts.max(axis=0) - nav_verts.min(axis=0) + 2 * margin
        self.W = int(np.ceil(span[0] / cell)) + 2   # cols = X
        self.H = int(np.ceil(span[1] / cell)) + 2   # rows = Y

        # Rasterise nav vertices
        free = np.zeros((self.H, self.W), dtype=bool)
        cols = np.round((nav_verts[:, 0] - self.origin[0]) / cell).astype(int)
        rows = np.round((nav_verts[:, 1] - self.origin[1]) / cell).astype(int)
        mask = (cols >= 0) & (cols < self.W) & (rows >= 0) & (rows < self.H)
        free[rows[mask], cols[mask]] = True

        # Dilation (1 cell) fills sub-cell gaps between adjacent nav points
        struct = np.ones((3, 3), dtype=bool)
        free_dilated = binary_dilation(free, structure=struct, iterations=1)

        # Erosion by robot radius adds obstacle clearance
        n_erode = max(1, int(np.round(robot_radius / cell)))
        self.free = binary_erosion(free_dilated, structure=struct,
                                   iterations=n_erode, border_value=False)

        self.free_before_erosion = free_dilated
        self.robot_radius = robot_radius

        # Distance transform: continuous distance to nearest obstacle (in cells)
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
        """Render the occupancy grid + planned path to an RGB numpy image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

        dist_m = self.dist_map * self.cell
        cost = 1.0 + self.PROXIMITY_WEIGHT / (dist_m + self.cell)
        cost[~self.free_before_erosion] = np.nan

        vmax = 1.0 + self.PROXIMITY_WEIGHT / (self.robot_radius + self.cell)
        extent = [
            self.origin[0], self.origin[0] + self.W * self.cell,
            self.origin[1], self.origin[1] + self.H * self.cell,
        ]
        im = ax.imshow(cost, origin="lower", extent=extent,
                       interpolation="nearest", cmap="RdYlGn_r",
                       vmin=1.0, vmax=vmax, zorder=1)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="A* step cost")

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
        if target_xy is not None:
            ax.plot(*target_xy, "r*", markersize=16, zorder=5, label="object")
            circle_obj = plt.Circle(target_xy, NAV_STOP_DIST,
                                    color="red", fill=False, linestyle=":",
                                    linewidth=1.0, zorder=4,
                                    label=f"stop={NAV_STOP_DIST:.2f}m")
            ax.add_patch(circle_obj)
        if robot_xy is not None:
            ax.plot(*robot_xy, "c^", markersize=10, zorder=6, label="robot")

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
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"  Debug image saved → {save_path}")

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w_px, h_px = fig.canvas.get_width_height()
        img = buf.reshape(h_px, w_px, 4)[..., :3].copy()
        plt.close(fig)
        return img

    def log_debug_grid(self, waypoints, start_xy, goal_xy,
                       target_xy=None, robot_xy=None, save_path=None) -> np.ndarray:
        """Render and optionally save the debug grid image."""
        return self.debug_image(waypoints, start_xy, goal_xy,
                                target_xy=target_xy, robot_xy=robot_xy,
                                save_path=save_path)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _w2g(self, xy: np.ndarray) -> tuple[int, int]:
        """World XY → (col, row)."""
        r = (xy - self.origin) / self.cell
        return int(round(r[0])), int(round(r[1]))

    def _g2w(self, col: int, row: int) -> np.ndarray:
        """(col, row) → world XY."""
        return self.origin + np.array([col, row], dtype=np.float64) * self.cell

    def _snap(self, col: int, row: int,
              grid: np.ndarray | None = None) -> tuple[int, int]:
        """Snap (col, row) to nearest True cell in `grid` (default: self.free)."""
        if grid is None:
            grid = self.free
        if 0 <= row < self.H and 0 <= col < self.W and grid[row, col]:
            return col, row
        free_rows, free_cols = np.where(grid)
        if len(free_rows) == 0:
            return col, row
        dists = (free_rows - row) ** 2 + (free_cols - col) ** 2
        idx = int(np.argmin(dists))
        return int(free_cols[idx]), int(free_rows[idx])

    # ── A* ────────────────────────────────────────────────────────────────────

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> list[np.ndarray]:
        """Plan a collision-free path from start to goal.

        Returns a list of XY waypoints (world frame).  Falls back to a
        single-waypoint direct path if A* finds no route.
        """
        sc, sr = self._snap(*self._w2g(start_xy), grid=self.free_before_erosion)
        gc, gr = self._snap(*self._w2g(goal_xy),  grid=self.free_before_erosion)

        if (sc, sr) == (gc, gr):
            return [goal_xy.copy()]

        SQRT2 = math.sqrt(2.0)
        h = lambda c, r: math.hypot(c - gc, r - gr)

        open_set: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (h(sc, sr), 0.0, sc, sr))
        g_score: dict[tuple[int, int], float] = {(sc, sr): 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        while open_set:
            _, g, c, r = heapq.heappop(open_set)

            if (c, r) == (gc, gr):
                path_cells = []
                node = (gc, gr)
                while node in came_from:
                    path_cells.append(node)
                    node = came_from[node]
                path_cells.append((sc, sr))
                path_cells.reverse()
                simplified = self._vis_simplify(path_cells)
                return [self._g2w(c_, r_) for c_, r_ in simplified]

            if g > g_score.get((c, r), float("inf")) + 1e-9:
                continue

            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = c + dc, r + dr
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    if not self.free_before_erosion[nr, nc]:
                        continue
                    base = SQRT2 if dc != 0 and dr != 0 else 1.0
                    dist_m = float(self.dist_map[nr, nc]) * self.cell
                    step = base * (1.0 + self.PROXIMITY_WEIGHT / (dist_m + self.cell))
                    ng = g + step
                    if ng < g_score.get((nc, nr), float("inf")):
                        g_score[(nc, nr)] = ng
                        came_from[(nc, nr)] = (c, r)
                        heapq.heappush(open_set, (ng + h(nc, nr), ng, nc, nr))

        print("  [NavGrid] A* found no path, falling back to direct navigation.")
        return [goal_xy.copy()]

    # ── path simplification ───────────────────────────────────────────────────

    def _los_ok(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Bresenham line-of-sight: every cell on the line from a→b must be
        inside self.free (the fully-clear zone, not the clearance zone)."""
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
        """Greedy visibility / string-pull: skip waypoints when LOS holds."""
        if len(path_cells) <= 2:
            return path_cells
        result = [path_cells[0]]
        i = 0
        while i < len(path_cells) - 1:
            j = len(path_cells) - 1
            while j > i + 1 and not self._los_ok(path_cells[i], path_cells[j]):
                j -= 1
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
    """
    qp = agent.robot.get_qpos().cpu().numpy().flatten().copy()
    yaw = qp[2]

    if abs(bearing) > ROT_THRESH:
        step = np.clip(bearing, -NAV_ROT_RAD_PER_STEP, NAV_ROT_RAD_PER_STEP)
        qp[2] = yaw + step
    else:
        import math as _math
        qp[0] += NAV_FWD_M_PER_STEP * _math.cos(yaw)
        qp[1] += NAV_FWD_M_PER_STEP * _math.sin(yaw)

    agent.robot.set_qpos(qp)
