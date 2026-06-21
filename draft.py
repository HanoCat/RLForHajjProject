import json
import copy

INPUT_JSON = "scene_coordinates.json"
OUTPUT_JSON = "scene_coordinates_fixed_barrier23.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    scene = json.load(f)

new_objects = []
barrier23 = None

for obj in scene["objects"]:
    if obj["id"] == 23 and obj["type"] == "polyline" and obj["label"].strip() == "barrier":
        barrier23 = obj
    else:
        new_objects.append(obj)

if barrier23 is None:
    raise ValueError("Barrier ID 23 not found.")

pts = barrier23["points"]

# Split barrier 23 into two separate barriers with a gap between them
barrier23_a = copy.deepcopy(barrier23)
barrier23_a["id"] = 2301
barrier23_a["points"] = pts[0:4]
barrier23_a["num_points"] = len(barrier23_a["points"])
barrier23_a["original_label"] = "barrier_split_from_23"

barrier23_b = copy.deepcopy(barrier23)
barrier23_b["id"] = 2302
barrier23_b["points"] = pts[6:11]
barrier23_b["num_points"] = len(barrier23_b["points"])
barrier23_b["original_label"] = "barrier_split_from_23"

new_objects.extend([barrier23_a, barrier23_b])

scene["objects"] = new_objects

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(scene, f, indent=2)

print("Saved:", OUTPUT_JSON)
print("Removed original barrier 23")
print("Added barrier 2301 and 2302")