import json
import matplotlib.pyplot as plt
from shapely import wkt
from shapely.affinity import rotate, translate
from shapely.validation import make_valid

from barrier_control import get_movable_barriers


ENV_JSON = "processed_environment.json"

# Edit these values manually while exploring.
# dx, dy are movement from original position in meters.
# angle is rotation relative to original orientation.
#  "movable_barrier_ids": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 2301, 2302],

CURRENT_BARRIER_POSES = {
    12:   {"x": 20.621, "y": 5.077,  "angle": -28.8},
    13:   {"x": 21.575, "y": 10.164, "angle": 11.1},
    14:   {"x": 24.277, "y": 13.779, "angle": -31.0},
    15:   {"x": 25.586, "y": 18.653, "angle": 22.6},
    16:   {"x": 25.985, "y": 32.366, "angle": -28.5},
    17:   {"x": 25.711, "y": 36.621, "angle": 38.7},
    18:   {"x": 50.500, "y": 37.612, "angle": -33.1},
    19:   {"x": 52.439, "y": 0.807,  "angle": -25.3},
    20:   {"x": 53.286, "y": 6.136,  "angle": 23.1},
    21:   {"x": 58.138, "y": 11.369, "angle": -47.5},
    22:   {"x": 59.023, "y": 16.663, "angle": 37.6},
    24:   {"x": 70.984, "y": 33.077, "angle": -26.5},
    2301: {"x": 52.682, "y": 33.765, "angle": 55.2},
    2302: {"x": 71.823, "y": 36.692, "angle": 25.4},
}

POSE_CONFIG = {
    12: { #done
        "closed": {"dx": -0.20, "dy":  -0.3, "angle": -1},
        "open":   {"dx": 0.0, "dy": 0.3, "angle": 150},
    },
    13: { #done
        "closed": {"dx": -0.30, "dy": 0.9, "angle": 10},
        "open":   {"dx": 0.0, "dy": -0.35, "angle": -150},
    },
    14: { #done
        "closed": {"dx":  0.35, "dy":  -0.60, "angle": 1},
        "open":   {"dx": 0.0, "dy": 0.0, "angle": 140},
    },
    15: { #done
        "closed": {"dx": -0.30, "dy":  0.70, "angle": -1},
        "open":   {"dx": 0.0, "dy": 0.0, "angle": -140},
    },

    16: {#done
        "closed": {"dx": 0.0, "dy": -0.75, "angle": 15},
        "open":   {"dx": -0.25, "dy":  0.15, "angle": -30},
    },
    17: {#done
        "closed": {"dx": 0.0, "dy": 0.75, "angle": -10},
        "open":   {"dx":  0.30, "dy": -0.40, "angle": -160},
    },

    18: {#done
        "closed": {"dx": 0.0, "dy": 0.20, "angle": 0},
        "open":   {"dx": -0.20, "dy": -0.25, "angle": -50},
    },
    19: {#done
        "closed": {"dx": 1.50, "dy": -0.60, "angle": -10},
        "open":   {"dx":  0.25, "dy":  0.20, "angle": 30},
    },
    20: { #done
        "closed": {"dx": 0.0, "dy": 0.25, "angle": 0},
        "open":   {"dx": -0.30, "dy":  0.20, "angle": -150},
    },
    22: { #done
        "closed": {"dx": 0.0, "dy": -0.25, "angle": 0},
        "open":   {"dx": -0.20, "dy":  0.0, "angle": 150},
    },
    21: { #done
        "closed": {"dx": 0.0, "dy": -0.55, "angle": 0},
        "open":   {"dx":  0.25, "dy":  0.10, "angle": -25},
    },
    24: { #done
        "closed": {"dx": 0.0, "dy": 0.90, "angle": -10},
        "open":   {"dx": -0.25, "dy": -0.25, "angle": 30},
    },
    2301: {#done
        "closed": {"dx": 0.0, "dy": -0.80, "angle": 30},
        "open":   {"dx": -0.25, "dy": -0.15, "angle": 140},
    },
    2302: {#done
        "closed": {"dx": 0.0, "dy": 0.90, "angle": -20},
        "open":   {"dx":  0.25, "dy": -0.20, "angle": -150},
    },
}

def apply_pose(poly, pose):
    rotated = rotate(
        poly,
        pose["angle"],
        origin="centroid",
        use_radians=False,
    )
    moved = translate(
        rotated,
        xoff=pose["dx"],
        yoff=pose["dy"],
    )
    return make_valid(moved.buffer(0))



with open(ENV_JSON, "r", encoding="utf-8") as f:
    env = json.load(f)

geometry = make_valid(wkt.loads(env["geometry_wkt"]["connected_geometry"]))
barriers = get_movable_barriers(env)

fig, ax = plt.subplots(figsize=(16, 8))

# Draw environment
x, y = geometry.exterior.xy
ax.fill(x, y, alpha=0.15)


# Draw configured poses
for bid, poses in POSE_CONFIG.items():
    if bid not in barriers:
        print(f"WARNING: barrier id {bid} not found.")
        continue

    original = barriers[bid]

    closed_poly = apply_pose(original, poses["closed"])
    open_poly = apply_pose(original, poses["open"])


    cx, cy = closed_poly.exterior.xy
    ox, oy = open_poly.exterior.xy

    ax.plot(cx, cy, linewidth=3, label=f"{bid} closed", color='red')
    ax.plot(ox, oy, linewidth=3, linestyle="--", label=f"{bid} open", color='green')


    print(
        f"Barrier {bid}:",
        "closed=", poses["closed"],
        "open=", poses["open"],

    )

ax.set_aspect("equal")
ax.set_title("Explore barrier closed/open poses: solid=closed, dashed=open")
plt.show()