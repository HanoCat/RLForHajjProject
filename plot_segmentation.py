import json
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon

IMAGE_PATH = "clean_seg1.png"
JSON_PATH = "scene_coordinates.json"

img = Image.open(IMAGE_PATH)

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

colors = {
    "walkable_area": "blue",
    "building_or_tent": "orange",
    "work_space": "cyan",
    "barrier": "red",
}

fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(img)

for obj in data["objects"]:
    label = obj["label"]
    obj_type = obj["type"]
    points = obj["points"]

    if label not in colors:
        continue

    if obj_type == "polygon":
        patch = Polygon(
            points,
            closed=True,
            fill=False,
            color=colors[label],
            edgecolor=colors[label],
            linewidth=2,
            label=label
        )
        ax.add_patch(patch)

    elif obj_type == "polyline":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=colors[label], linewidth=2, label=label)

ax.set_xlim(0, img.width)
ax.set_ylim(img.height, 0)
ax.axis("off")

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc="upper right")

plt.tight_layout()
plt.show()