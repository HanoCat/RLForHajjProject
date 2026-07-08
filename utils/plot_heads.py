import os
import json
import matplotlib.pyplot as plt
from shapely import wkt

from config.scenario_config import SCENARIO
from utils.simulation_utils import load_environment
from barrier_control import clean_geom


HEAD_FILE = "../CrowdCounting-P2PNet/predicted_head_points.json"
OUTPUT_DIR = "../CrowdCounting-P2PNet/paper_figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "head_detections_red.png")

SCALE = 0.05          # meter per pixel
IMAGE_HEIGHT_PX = 780 # because env height ≈ 39m / 0.05


def plot_polygon(ax, geom, facecolor="lightgray", edgecolor="black", alpha=0.35):
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]

    for g in geoms:
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=0.8)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, env = load_environment(SCENARIO["env_json"])

    geometry = clean_geom(
        wkt.loads(env["geometry_wkt"]["simulation_geometry"])
    )

    with open(HEAD_FILE, "r") as f:
        data = json.load(f)

    agents = data["agents"]

    xs = []
    ys = []

    for a in agents:
        x_m = a["x"] * SCALE
        y_m = (IMAGE_HEIGHT_PX - a["y"]) * SCALE

        xs.append(x_m)
        ys.append(y_m)

    fig, ax = plt.subplots(figsize=(14, 7))

    plot_polygon(ax, geometry)

    ax.scatter(
        xs,
        ys,
        s=8,
        c="red",
        alpha=0.75,
        label="Detected heads"
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Detected head points over simulation environment")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print("Number of detected heads:", len(agents))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()