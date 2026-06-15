#!/usr/bin/env python3
"""Count skewed (tilted) railroad-crossing polygons in a YOLO label set.

Labels are normalized 4-corner polygons: `class x1 y1 x2 y2 x3 y3 x4 y4`.
A polygon is "skewed" when its edges are rotated away from horizontal/vertical.
For each of the 4 edges we measure how far its angle sits from the nearest
0deg/90deg axis; the polygon's skew is the largest of those four deviations
(an upright rectangle = 0deg, a rectangle rotated by t = t).

Usage:
    python3 count_skewed_railroad.py
    python3 count_skewed_railroad.py --labels <dir> --threshold 5
    python3 count_skewed_railroad.py --class-id 0 --top 15
"""

import argparse
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
DEFAULT_LABELS = str(SCRIPT_DIR / "datasets/yolo_pl_test/labels")
DEFAULT_THRESHOLD_DEG = 5.0


def edge_skew_deg(p):
    """Max deviation (degrees) of the 4 polygon edges from horizontal/vertical.

    p is [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]. Returns None if every edge is
    degenerate (zero length).
    """
    worst = None
    for i in range(4):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % 4]
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            continue
        ang = abs(math.degrees(math.atan2(dy, dx))) % 90.0  # fold to 0..90
        dev = min(ang, 90.0 - ang)                          # distance to 0 or 90
        worst = dev if worst is None else max(worst, dev)
    return worst


def scan(labels_dir, class_id):
    """Return (polygons, n_nonpolygon) for the given class.

    polygons    -> list of (file_name, [(x,y)*4], skew_deg)
    n_nonpolygon-> count of class_id labels that are NOT 4-corner polygons
                   (e.g. 5-field `class x y w h` centroid labels carry no
                   orientation, so skew cannot be measured for them).
    """
    polygons = []
    n_nonpolygon = 0
    for txt in sorted(Path(labels_dir).glob("*.txt")):
        for line in txt.read_text().splitlines():
            f = line.split()
            if len(f) < 3 or int(f[0]) != class_id:
                continue
            if len(f) != 9:
                n_nonpolygon += 1
                continue
            corners = [(float(f[i]), float(f[i + 1])) for i in (1, 3, 5, 7)]
            skew = edge_skew_deg(corners)
            if skew is not None:
                polygons.append((txt.name, corners, skew))
    return polygons, n_nonpolygon


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="Directory of YOLO 4-corner .txt label files.")
    ap.add_argument("--class-id", type=int, default=0,
                    help="Class id to inspect (default 0 = railroad-crossing).")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_DEG,
                    help="Skew angle (degrees) above which a polygon counts as skewed.")
    ap.add_argument("--top", type=int, default=10,
                    help="Show this many most-skewed examples.")
    args = ap.parse_args()

    if not Path(args.labels).is_dir():
        raise SystemExit(f"Labels directory not found: {args.labels}")

    cname = (DEFAULT_CLASSES[args.class_id]
             if 0 <= args.class_id < len(DEFAULT_CLASSES) else str(args.class_id))

    rows, n_nonpolygon = scan(args.labels, args.class_id)
    total = len(rows)
    if total == 0:
        if n_nonpolygon:
            print(f"Class {args.class_id} ({cname}) in {args.labels}: "
                  f"{n_nonpolygon} labels found, but none are 4-corner polygons "
                  f"(centroid-format labels carry no orientation — skew is undefined).")
        else:
            print(f"No class {args.class_id} ({cname}) labels found in {args.labels}")
        return

    skewed = [r for r in rows if r[2] > args.threshold]
    skews = [r[2] for r in rows]

    print(f"Class {args.class_id} ({cname}) in {args.labels}")
    print(f"  polygons:        {total}" +
          (f"  (+{n_nonpolygon} centroid-format, skipped)" if n_nonpolygon else ""))
    print(f"  skewed > {args.threshold:g}deg:   {len(skewed)} ({100 * len(skewed) / total:.1f}%)")
    print(f"  skew angle deg:  min={min(skews):.2f}  median={sorted(skews)[len(skews)//2]:.2f}  max={max(skews):.2f}")

    # distribution by degree band
    bands = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 45)]
    print("  distribution:")
    for lo, hi in bands:
        n = sum(1 for s in skews if lo <= s < hi)
        print(f"    {lo:>2}-{hi:<2}deg: {n:>4} ({100 * n / total:>5.1f}%)")

    if skewed:
        print(f"\n  top {min(args.top, len(skewed))} most skewed:")
        for name, _, skew in sorted(skewed, key=lambda r: -r[2])[:args.top]:
            print(f"    {skew:6.2f}deg  {name}")


if __name__ == "__main__":
    main()
