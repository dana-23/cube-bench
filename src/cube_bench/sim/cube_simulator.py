"""Simulation, state conversion, and rendering for a 3x3 Rubik's Cube."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import kociemba
import matplotlib.pyplot as plt
import numpy as np
import pycuber as pc  # type: ignore – external dependency
from kociemba.pykociemba.facecube import FaceCube
from PIL import Image

import cube_bench.optimal.solver as sv

_ORACLE_LOCK = Lock()

# Configuration helpers
@dataclass(frozen=True)
class Palette:
    """Maps logical cube colours to RGB-255 triples."""

    colour_to_rgb: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "white": (255, 255, 255),
            "yellow": (255, 255, 0),
            "orange": (255, 128, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
        }
    )

    def __getitem__(self, colour: str) -> Tuple[int, int, int]:
        return self.colour_to_rgb[colour.lower()]

@dataclass(frozen=True)
class NetLayout:
    """Pre-computed positions (top-left y, x) for each cube face in the 2-D net."""

    face_px: int
    face_gap: int
    # Derived from face_px/face_gap in __post_init__; excluded from eq/hash so the
    # frozen dataclass stays hashable (a dict field would make it unhashable).
    positions: Dict[str, Tuple[int, int]] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "positions", self._compute_positions())

    def _compute_positions(self) -> Dict[str, Tuple[int, int]]:
        s, g = self.face_px, self.face_gap
        return {
            "U": (0, s + g),
            "L": (s + g, 0),
            "F": (s + g, s + g),
            "R": (s + g, 2 * (s + g)),
            "B": (s + g, 3 * (s + g)),
            "D": (2 * (s + g), s + g),
        }

    def canvas_shape(self) -> Tuple[int, int]:
        """Height, width of the full unfolded cube canvas in pixels."""
        h = 3 * self.face_px + 2 * self.face_gap
        w = 4 * self.face_px + 3 * self.face_gap
        return h, w


# Core class
@lru_cache(maxsize=262144)
def _co_eo_from_facelets(facelets: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Convert URFDLB facelets to corner twists and edge flips."""
    cc = FaceCube(facelets).toCubieCube()
    return tuple(int(x) for x in cc.co[:8]), tuple(int(x) for x in cc.eo[:12])


@lru_cache(maxsize=262144)
def _solve_with_oracle(facelets: str) -> str:
    """Optimal solution for a facelet string; pure in its argument, so memoized."""
    with _ORACLE_LOCK:
        return sv.solve(facelets)


class VirtualCube:
    """A lightweight facade around *pycuber*'s `Cube` with rendering helpers."""

    #: Human‑friendly labels for annotation – edit freely.
    FACE_LABELS = {"U": "Top", "L": "Left", "F": "Front", "R": "Right", "B": "Back", "D": "Down"}

    #: Faces in the order they appear in the unfolded net
    FACE_ORDER = ["U", "L", "F", "R", "B", "D"]

    #: Basic and double turns the cube understands (no slice/wide moves)
    AVAILABLE_MOVES: Tuple[str, ...] = (
        "R", "L", "U", "D", "F", "B",
        "R'", "L'", "U'", "D'", "F'", "B'",
        "R2", "L2", "U2", "D2", "F2", "B2",
    )

    _COLOR_ALIASES = {
        "w": "white", "white": "white",
        "y": "yellow","yellow":"yellow",
        "o": "orange","orange":"orange",
        "r": "red",   "red":   "red",
        "g": "green", "green": "green",
        "b": "blue",  "blue":  "blue",
    }

    def _canon(self, name: str) -> str:
        return self._COLOR_ALIASES.get(str(name).lower().strip(), str(name).lower().strip())

    # Construction & simple helpers

    def __init__(self, cube: Optional[pc.Cube] = None) -> None:
        self._cube: pc.Cube = cube if cube is not None else pc.Cube()
        self._palette = Palette()
        # Most recently applied scramble.
        self.formula: Optional[pc.Formula] = None

    @property
    def raw(self) -> pc.Cube:
        """The underlying *pycuber* cube, for callers that need its API directly."""
        return self._cube

    def __str__(self) -> str:
        return self._cube.__str__()

    def is_solved(self) -> bool:
        """True if each face is uniform (scheme-agnostic)."""

        def center_matching():
            for f in "ULFRBD":
                face = self._cube.get_face(f)
                c0 = face[1][1].colour
                if any(sq.colour != c0 for row in face for sq in row):
                    return False

            return True

        def lazy_matching():
            solved_cube = pc.Cube()
            return str(self._cube) == str(solved_cube)

        return center_matching() or lazy_matching()

    def clone(self) -> "VirtualCube":
        """Return an independent copy of this VirtualCube."""
        return VirtualCube(self._cube.copy())

    def get_distance(self) -> int:
        """Optimal solution length in half-turn metric (0 when solved)."""
        if self.is_solved():
            return 0

        s54 = self.to_kociemba()
        solution = _solve_with_oracle(s54)
        solution = solution.split(" ")

        distance = solution[-1]
        distance = re.search(r"\d+", distance)

        return int(distance.group())

    def corner_orientations(self) -> list[int]:
        """Per-corner twist values derived from the current facelets."""
        co, _ = _co_eo_from_facelets(self.to_kociemba())
        return list(co)

    def edge_orientations(self) -> list[int]:
        """Per-edge flip values derived from the current facelets."""
        _, eo = _co_eo_from_facelets(self.to_kociemba())
        return list(eo)

    def scramble(
        self,
        random_seed: int = 69,
        n_moves: int = 20,
        max_tries: int = 50,
        *,
        exact_depth: bool = False,
    ) -> pc.Formula:
        """Apply a seeded scramble, optionally requiring exact oracle depth."""
        if n_moves < 0:
            raise ValueError("n_moves must be non-negative")
        if max_tries < 1:
            raise ValueError("max_tries must be positive")
        if exact_depth and n_moves > 20:
            raise ValueError("Exact Rubik's Cube distance cannot exceed 20 in HTM")

        rng = np.random.default_rng(random_seed)

        def face_of(move: str) -> str:
            return move[0]

        def sample_moves() -> list[str]:
            seq = []
            last_face = None
            for _ in range(n_moves):
                candidates = [m for m in self.AVAILABLE_MOVES if face_of(m) != last_face] \
                            if last_face else list(self.AVAILABLE_MOVES)
                m = candidates[rng.integers(len(candidates))]
                seq.append(m)
                last_face = face_of(m)
            return seq

        for _ in range(max_tries):
            moves = sample_moves()
            formula = pc.Formula(moves)
            self._cube(formula)
            accepted = self.get_distance() == n_moves if exact_depth else not self.is_solved()
            if accepted:
                self.formula = formula.copy()
                return self.formula.copy()
            self._cube(formula.copy().reverse())

        requirement = f"exact depth {n_moves}" if exact_depth else "a non-solved state"
        raise RuntimeError(
            f"Could not generate {requirement} after {max_tries} attempts "
            f"(seed={random_seed}, n_moves={n_moves})"
        )

    def apply(self, moves: str) -> None:
        """Apply a move sequence given in standard notation (e.g. "R U R' U'")."""
        self._cube(moves)

    def solve(self):
        """Return an optimal solution in standard notation ("" when solved)."""
        if self.is_solved():
            return ""

        s54 = self.to_kociemba()
        solution = _solve_with_oracle(s54)
        solution = solution.split(" ")

        for i, op in enumerate(solution):
            if op.endswith("1"):
                solution[i] = op[0]

            elif op.endswith("3"):
                solution[i] = op[0] + "'"

        optimal_solution = " ".join(solution[:-1])

        return optimal_solution

    def front_face(self) -> List[List[str]]:
        """Return the current colours of the *Front* face (3x3 list)."""
        face = self._cube.get_face("F")
        return [[sq.colour for sq in row] for row in face]

    def reset(self):
        """Reset the cube to its solved state."""
        self._cube: pc.Cube = pc.Cube()

    def to_kociemba(self, net: str | None = None) -> str:  # pylint: disable=too-many-branches
        """Export URFDLB facelets, using current centres to support isomorphic recolours."""
        color_to_face = {
            str(self._cube.get_face(f)[1][1].colour).lower(): f
            for f in "URFDLB"
        }
        if len(color_to_face) != 6:
            raise ValueError("Center colors must be unique; current scheme appears invalid.")

        def _token_to_faceletter(tok: str) -> str:
            col = self._canon(tok)
            try:
                return color_to_face[col]
            except KeyError as e:
                raise ValueError(f"Unknown sticker color token '{tok}' (-> '{col}') for current centers.") from e

        out: list[str] = []

        if net is None:
            for f in "URFDLB":
                face = self._cube.get_face(f)
                for r in range(3):
                    for c in range(3):
                        col = str(face[r][c].colour).lower()
                        try:
                            out.append(color_to_face[col])
                        except KeyError as e:
                            raise ValueError(f"Sticker color '{col}' not present in center mapping.") from e
        else:
            rows = net.strip().splitlines()
            token_re = re.compile(r"\[([a-zA-Z]+)\]")

            faces_tokens: Dict[str, List[str]] = {k: [] for k in "ULFRBD"}
            for row_idx, row in enumerate(rows):
                tokens = token_re.findall(row)
                if not tokens:
                    continue
                if row_idx <= 2:
                    faces_tokens["U"].extend(tokens)
                elif 3 <= row_idx <= 5:
                    if len(tokens) >= 12:
                        faces_tokens["L"].extend(tokens[0:3])
                        faces_tokens["F"].extend(tokens[3:6])
                        faces_tokens["R"].extend(tokens[6:9])
                        faces_tokens["B"].extend(tokens[9:12])
                else:
                    faces_tokens["D"].extend(tokens)

            for f in "URFDLB":
                toks = faces_tokens[f]
                if len(toks) != 9:
                    raise ValueError(f"Face '{f}' does not have 9 tokens in provided net.")
                out.extend(_token_to_faceletter(t) for t in toks)

        assert len(out) == 54, f"Expected 54 facelets, got {len(out)}."
        return "".join(out)

    def from_kociemba(self, state54: str | None = None) -> str:
        """Build an ASCII net from URFDLB facelets in the current centre colour scheme."""
        if not state54:
            state54 = self.to_kociemba()

        moves = kociemba.solve(state54)

        temp = pc.Cube()
        temp(pc.Formula(moves).reverse())

        default_center_by_face = {
            f: str(pc.Cube().get_face(f)[1][1].colour).lower()
            for f in "URFDLB"
        }
        current_center_by_face = {
            f: str(self._cube.get_face(f)[1][1].colour).lower()
            for f in "URFDLB"
        }
        recolor_map = {
            default_center_by_face[f]: current_center_by_face[f]
            for f in "URFDLB"
        }
        if len(set(recolor_map.values())) != 6:
            # Rendering tolerates non-bijective maps; mutation validates more strictly.
            pass

        for f in "ULFRBD":
            face = temp.get_face(f)
            for r in range(3):
                for c in range(3):
                    sq = face[r][c]
                    src = str(sq.colour).lower()
                    tgt = recolor_map.get(src, src)
                    sq.colour = tgt

        # Copying rebuilds pycuber's colour-keyed internal containers.
        temp = temp.copy()

        return str(temp)

    # Rendering
    def render(self, *, cell_size: int = 60, sticker_border: int = 2,
               face_gap: int = 40, file_path: Optional[Union[str, Path]] = None,
               dpi: int = 100, add_labels: bool = True):
        """Render the cube net as a PIL image, or as a path when ``file_path`` is supplied."""
        canvas, layout = self._build_canvas(
            cell_size=cell_size,
            sticker_border=sticker_border,
            face_gap=face_gap,
        )

        if add_labels or file_path is not None:
            fig = self._canvas_to_figure(canvas, layout, dpi=dpi, add_labels=add_labels)
            try:
                if file_path:
                    file_path = Path(file_path)
                    if not file_path.suffix:
                        file_path = file_path.with_suffix(".png")
                    fig.savefig(
                        file_path,
                        dpi=dpi,
                        bbox_inches="tight",
                        pad_inches=0.1,
                        facecolor=fig.get_facecolor(),
                    )
                    return Path(file_path)

                fig.canvas.draw()
                rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()

                # Retina buffers can be larger than the figure's logical dimensions.
                scale = int(np.sqrt(rgba.size / (w * h * 4)))
                if scale > 1:
                    w, h = w * scale, h * scale

                canvas = rgba.reshape(h, w, 4)[..., :3].copy()
            finally:
                plt.close(fig)

        if Image is None:
            raise RuntimeError("Pillow not installed; cannot return PIL image.")
        return Image.fromarray(canvas, mode="RGB")

    def to_image(self, file_path: Optional[Union[str, Path]] = None, **kwargs):
        """Return a PIL image, or a path when ``file_path`` is supplied."""
        return self.render(file_path=file_path, **kwargs)




    # Canvas internals
    def _build_canvas(self, *, cell_size: int, sticker_border: int, face_gap: int):
        layout = NetLayout(face_px=3 * cell_size, face_gap=face_gap)
        h, w = layout.canvas_shape()
        canvas = np.full((h, w, 3), 127, dtype=np.uint8)
        for face_key in self.FACE_ORDER:
            self._paint_face(
                canvas,
                face_key,
                origin=layout.positions[face_key],
                cell_size=cell_size,
                sticker_border=sticker_border,
            )
        return canvas, layout

    def _canvas_to_figure(self, canvas: np.ndarray, layout: NetLayout, *, dpi: int, add_labels: bool):
        canvas_h, canvas_w = canvas.shape[:2]
        fig_w_in, fig_h_in = canvas_w / dpi, canvas_h / dpi
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        grey = (125 / 255,) * 3
        fig.patch.set_facecolor(grey)
        ax.set_position([0, 0, 1, 1])
        ax.imshow(canvas, interpolation="nearest")
        ax.axis("off")
        if add_labels:
            for face_key in self.FACE_ORDER:
                y0, x0 = layout.positions[face_key]
                ax.text(
                    x0 + layout.face_px / 2,
                    y0 - 6,
                    self.FACE_LABELS[face_key],
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="white",
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.6, "linewidth": 0},
                )
        fig.tight_layout(pad=0)
        return fig

    def _paint_face(self, canvas: np.ndarray, face_key: str, *, origin: Tuple[int, int],
                    cell_size: int, sticker_border: int) -> None:
        """Blit a single 3x3 face onto the *canvas* at *origin*."""
        y0, x0 = origin
        face_px = 3 * cell_size
        face_img = np.zeros((face_px, face_px, 3), dtype=np.uint8)

        face_grid = self._cube.get_face(face_key)
        for r in range(3):
            for c in range(3):
                rgb = self._palette[str(face_grid[r][c].colour)]
                r_lo, r_hi = r * cell_size + sticker_border, (r + 1) * cell_size - sticker_border
                c_lo, c_hi = c * cell_size + sticker_border, (c + 1) * cell_size - sticker_border
                face_img[r_lo:r_hi, c_lo:c_hi] = rgb

        canvas[y0 : y0 + face_px, x0 : x0 + face_px] = face_img


def _demo() -> None:
    """Write a 10-move scramble to cube_scramble.png."""
    cube = VirtualCube()
    cube.scramble(n_moves=10)
    cube.to_image("cube_scramble.png")


if __name__ == "__main__":
    _demo()
