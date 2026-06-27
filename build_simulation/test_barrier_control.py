# debug_barrier_mapping.py

import json
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.affinity import rotate
from shapely.validation import make_valid

from barrier_control import get_movable_barriers

with open("../processed_environment.json", "r", encoding="utf-8") as f:
    env = json.load(f)

geometry = make_valid(wkt.loads(env["geometry_wkt"]["connected_geometry"]))
barriers = get_movable_barriers(env)

fig, ax = plt.subplots(figsize=(16, 8))

x, y = geometry.exterior.xy
ax.fill(x, y, alpha=0.15)

for bid, poly in barriers.items():
    # original barrier
    ox, oy = poly.exterior.xy
    ax.plot(ox, oy, linewidth=2)

    # rotated barrier, exaggerated
    rotated = rotate(poly, 20, origin="centroid", use_radians=False)
    rx, ry = rotated.exterior.xy
    ax.plot(rx, ry, linestyle="--", linewidth=2)

    c = poly.centroid
    ax.text(c.x, c.y, str(bid), fontsize=11, weight="bold")

    print(
        bid,
        "| centroid:",
        round(c.x, 2),
        round(c.y, 2),
        "| bounds:",
        tuple(round(v, 2) for v in poly.bounds),
    )

ax.set_aspect("equal")
ax.set_title("Movable barrier ID mapping: solid=original, dashed=rotated +80°")
plt.show()

print()


import json
from pathlib import Path

import matplotlib.pyplot as plt
from shapely import wkt
from shapely.validation import make_valid

from barrier_control import apply_barrier_actions, rotate_barriers

BASE_DIR = Path(__file__).resolve().parent
ENV_JSON = BASE_DIR / "processed_environment.json"
OUT_IMAGE = BASE_DIR / "barrier_control_test.png"

with open(ENV_JSON, "r", encoding="utf-8") as f:
    env = json.load(f)

base_geometry = make_valid(wkt.loads(env["geometry_wkt"]["connected_geometry"]))

zero_actions = {bid: 0.0 for bid in env["metadata"]["movable_barrier_ids"]}
zero_geometry = apply_barrier_actions(env, zero_actions)

# A conservative test action set. Some large combinations can intentionally split
# the map into disconnected polygons, which JuPedSim should not receive directly.
sample_actions = {
    12: 30.0,
    14: 15.0,
    18: 30.0,
    21: 15.0,
    24: 30.0,
    2301: 30.0,
}
controlled_geometry = apply_barrier_actions(env, sample_actions)
rotated = rotate_barriers(env, sample_actions)

print("Base geometry valid:", base_geometry.is_valid)
print("Zero-action geometry valid:", zero_geometry.is_valid)
print("Controlled geometry valid:", controlled_geometry.is_valid)
print("Base area:", round(base_geometry.area, 4))
print("Zero-action area:", round(zero_geometry.area, 4))
print("Controlled area:", round(controlled_geometry.area, 4))
print("Base vs zero symmetric difference area:", round(base_geometry.symmetric_difference(zero_geometry).area, 8))
print("Controlled holes:", len(controlled_geometry.interiors))

fig, ax = plt.subplots(figsize=(14, 7))

x, y = controlled_geometry.exterior.xy
ax.fill(x, y, alpha=0.15)

for interior in controlled_geometry.interiors:
    ix, iy = interior.xy
    ax.plot(ix, iy, linewidth=1)

for bid, poly in rotated.items():
    bx, by = poly.exterior.xy
    ax.plot(bx, by, linewidth=2)
    c = poly.centroid
    ax.text(c.x, c.y, str(bid), fontsize=8)

ax.set_aspect("equal")
ax.set_title("Barrier control test: rotated movable barriers")
plt.show()