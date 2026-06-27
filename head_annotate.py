import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "CrowdCounting-P2PNet/clean_frame.png"
SCENE_JSON = "scene_coordinates_with_crowd.json"
P2PNET_JSON = "p2pnet_points.json"
OUTPUT_JSON = "hybrid_agents_p2pnet.json"

DENSE_MULTIPLIER = 4.0
MIN_DISTANCE = 5

img = cv2.imread(IMAGE_PATH)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

with open(SCENE_JSON, "r", encoding="utf-8") as f:
    scene = json.load(f)

with open(P2PNET_JSON, "r", encoding="utf-8") as f:
    detected = json.load(f)["agents"]

print("P2PNet detected agents:", len(detected))


def make_mask(label):
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj in scene["objects"]:
        if obj["type"] == "polygon" and obj["label"] == label:
            pts = np.array(obj["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def keep_inside_mask(points, mask, source):
    kept = []
    for p in points:
        x = int(round(p["x"]))
        y = int(round(p["y"]))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            kept.append({
                "x": float(p["x"]),
                "y": float(p["y"]),
                "source": source,
                "score": float(p.get("score", 1.0))
            })
    return kept


def generate_spaced_points(mask, target_count, min_distance=5):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    synthetic = []
    attempts = 0
    max_attempts = target_count * 200
    min_dist_sq = min_distance ** 2

    while len(synthetic) < target_count and attempts < max_attempts:
        attempts += 1

        idx = random.randrange(len(xs))
        x = float(xs[idx])
        y = float(ys[idx])

        too_close = False
        for p in synthetic:
            if (x - p["x"])**2 + (y - p["y"])**2 < min_dist_sq:
                too_close = True
                break

        if not too_close:
            synthetic.append({
                "x": x,
                "y": y,
                "source": "synthetic_dense",
                "score": None
            })

    return synthetic


sparse_mask = make_mask("crowd_sparse")
dense_mask = make_mask("crowd_dense")

real_sparse_agents = keep_inside_mask(detected, sparse_mask, "p2pnet_sparse")
real_dense_detected = keep_inside_mask(detected, dense_mask, "p2pnet_dense")

sparse_area = np.count_nonzero(sparse_mask)
dense_area = np.count_nonzero(dense_mask)

sparse_density = len(real_sparse_agents) / sparse_area if sparse_area > 0 else 0
target_dense_total = int(dense_area * sparse_density * DENSE_MULTIPLIER)

needed_synthetic_dense = max(0, target_dense_total - len(real_dense_detected))

synthetic_dense_agents = generate_spaced_points(
    dense_mask,
    needed_synthetic_dense,
    min_distance=MIN_DISTANCE
)

agents = []
all_points = real_sparse_agents + real_dense_detected + synthetic_dense_agents

for i, p in enumerate(all_points, start=1):
    agents.append({
        "id": i,
        "x": p["x"],
        "y": p["y"],
        "source": p["source"],
        "score": p["score"]
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"agents": agents}, f, indent=2)

print("P2PNet sparse:", len(real_sparse_agents))
print("P2PNet dense:", len(real_dense_detected))
print("Target dense total:", target_dense_total)
print("Synthetic dense added:", len(synthetic_dense_agents))
print("Total agents:", len(agents))
print("Saved:", OUTPUT_JSON)

plt.figure(figsize=(16, 8))
plt.imshow(rgb)

for source, size in [
    ("p2pnet_sparse", 6),
    ("p2pnet_dense", 5),
    ("synthetic_dense", 4),
]:
    xs = [p["x"] for p in agents if p["source"] == source]
    ys = [p["y"] for p in agents if p["source"] == source]
    plt.scatter(xs, ys, s=size, label=f"{source}: {len(xs)}")

plt.axis("off")
plt.legend()
plt.title(f"Hybrid P2PNet + synthetic agents: {len(agents)}")
plt.show()