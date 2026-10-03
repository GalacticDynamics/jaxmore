# /// script
# requires-python = ">=3.11"
# dependencies = ["resvg-py"]
# ///
"""Draw the jaxmore logo: JAX's, its J's stem crossed into a big +.

JAX's logo builds the letters J, A and X from rhombi on a triangular lattice.
jaxmore's is the same logo with the J's stem crossed, on the lattice, by a bar
through the A, so the stem reads as a big +, in dark grey. The shapes are
vector, so the logo is written as an SVG, sharp at any size; for a bitmap, name
a .png and give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
from pathlib import Path

# JAX's logo, from images/jax_logo.svg in https://github.com/jax-ml/jax, under the
# Apache License 2.0: each rhombus or triangle as (fill, points), drawn in order.
# The JAX name and logo are Google's.
JAX = [
    ("#5e97f6", "50.5 130.4 25.5 173.71 75.5 173.71 100.5 130.4"),
    ("#5e97f6", "0.5 217.01 25.5 173.71 75.5 173.71 50.5 217.01"),
    ("#5e97f6", "125.5 173.71 75.5 173.71 50.5 217.01 100.5 217.01"),
    ("#5e97f6", "175.5 173.71 125.5 173.71 100.5 217.01 150.5 217.01"),
    ("#5e97f6", "150.5 130.4 125.5 173.71 175.5 173.71 200.5 130.4"),
    ("#5e97f6", "175.5 87.1 150.5 130.4 200.5 130.4 225.5 87.1"),
    ("#5e97f6", "200.5 43.8 175.5 87.1 225.5 87.1 250.5 43.8"),
    ("#5e97f6", "225.5 0.5 200.5 43.8 250.5 43.8 275.5 0.5"),
    ("#2a56c6", "0.5 217.01 25.5 260.31 75.5 260.31 50.5 217.01"),
    ("#2a56c6", "125.5 260.31 75.5 260.31 50.5 217.01 100.5 217.01"),
    ("#2a56c6", "175.5 260.31 125.5 260.31 100.5 217.01 150.5 217.01"),
    ("#00796b", "200.5 217.01 175.5 173.71 150.5 217.01 175.5 260.31"),
    ("#00796b", "250.5 130.4 225.5 87.1 200.5 130.4"),
    ("#00796b", "250.5 43.8 225.5 87.1 250.5 130.4 275.5 87.1"),
    ("#3367d6", "125.5 173.71 100.5 130.4 75.5 173.71"),
    ("#26a69a", "250.5 130.4 200.5 130.4 175.5 173.71 225.5 173.71"),
    ("#26a69a", "300.5 130.4 250.5 130.4 225.5 173.71 275.5 173.71"),
    ("#9c27b0", "350.5 43.8 325.5 0.5 300.5 43.8 325.5 87.1"),
    ("#9c27b0", "375.5 87.1 350.5 43.8 325.5 87.1 350.5 130.4"),
    ("#9c27b0", "400.5 130.4 375.5 87.1 350.5 130.4 375.5 173.71"),
    ("#9c27b0", "425.5 173.71 400.5 130.4 375.5 173.71 400.5 217.01"),
    ("#9c27b0", "450.5 217.01 425.5 173.71 400.5 217.01 425.5 260.31"),
    ("#9c27b0", "425.5 0.5 400.5 43.8 425.5 87.1 450.5 43.8"),
    ("#9c27b0", "375.5 87.1 400.5 43.8 425.5 87.1 400.5 130.4"),
    ("#9c27b0", "350.5 130.4 325.5 173.71 350.5 217.01 375.5 173.71"),
    ("#9c27b0", "325.5 260.31 300.5 217.01 325.5 173.71 350.5 217.01"),
    ("#6a1b9a", "275.5 260.31 250.5 217.01 300.5 217.01 325.5 260.31"),
    ("#00695c", "225.5 173.71 175.5 173.71 200.5 217.01 250.5 217.01"),
    ("#00695c", "275.5 173.71 225.5 173.71 250.5 217.01"),
    ("#00695c", "275.5 87.1 300.5 130.4 350.5 130.4 325.5 87.1"),
    ("#00695c", "300.5 43.8 250.5 43.8 275.5 87.1 325.5 87.1"),
    ("#00695c", "425.5 260.31 400.5 217.01 350.5 217.01 375.5 260.31"),
    ("#00695c", "375.5 173.71 350.5 217.01 400.5 217.01"),
    ("#ea80fc", "325.5 0.5 275.5 0.5 250.5 43.8 300.5 43.8"),
    ("#ea80fc", "325.5 173.71 275.5 173.71 250.5 217.01 300.5 217.01"),
    ("#ea80fc", "350.5 130.4 300.5 130.4 275.5 173.71 325.5 173.71"),
    ("#ea80fc", "425.5 0.5 375.5 0.5 350.5 43.8 400.5 43.8"),
    ("#ea80fc", "375.5 87.1 350.5 43.8 400.5 43.8"),
]
EDGE = "#dce0df"  # the light line between the cells
PLUS = "#3c4043"  # dark grey, so the + shows on dark backgrounds too

ROW = 43.3  # the lattice's row height; its cells are 50 wide
# The +: the J's stem on rows 0-4, and a bar along row 2 two cells each way. A
# cell is named by its top edge's left end and its row.
STEM = [(225.5, 0), (200.5, 1), (175.5, 2), (150.5, 3), (125.5, 4)]
BAR = [(75.5, 2), (125.5, 2), (225.5, 2), (275.5, 2)]


def cell(a: float, row: int) -> str:
    """Return the points of the rhombus with top edge [a, a + 50] on ``row``."""
    y0, y1 = 0.5 + ROW * row, 0.5 + ROW * (row + 1)
    return f"{a:g} {y0:g} {a - 25:g} {y1:g} {a + 25:g} {y1:g} {a + 50:g} {y0:g}"


def top_left(points: str) -> tuple[float, int]:
    """Return a cell's top edge's left end, and its row."""
    xy = [float(v) for v in points.split()]
    top = min(xy[1::2])
    left = min(x for x, y in zip(xy[0::2], xy[1::2], strict=True) if y == top)
    return left, round((top - 0.5) / ROW)


def svg() -> str:
    """Return the logo as SVG text."""
    stem = set(STEM)
    shapes = [
        (PLUS if top_left(points) in stem else fill, points) for fill, points in JAX
    ]
    shapes += [(PLUS, cell(a, row)) for a, row in BAR]  # over the A
    polygons = "\n".join(
        f'    <polygon fill="{fill}" points="{points}"/>' for fill, points in shapes
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1 -1 453 263"'
        ' width="512" height="297">\n'
        f'  <g stroke="{EDGE}" stroke-linejoin="round">\n{polygons}\n  </g>\n</svg>\n'
    )


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument("--size", type=int, default=512, help="pixels wide, for a PNG")
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        import resvg_py  # noqa: PLC0415  # only a PNG needs a renderer

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
