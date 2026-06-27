import zipfile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

INPUT_ZIP = "crowd_scene_manual_segmentation_img.zip"
OUTPUT_JSON = "scene_coordinates_with_crowd.json"

def parse_points(points_str):
    points = []
    for pair in points_str.split(";"):
        x, y = pair.split(",")
        points.append([float(x), float(y)])
    return points

with zipfile.ZipFile(INPUT_ZIP, "r") as z:
    xml_files = [name for name in z.namelist() if name.endswith(".xml")]
    if not xml_files:
        raise FileNotFoundError("No XML file found inside the ZIP.")

    xml_name = xml_files[0]
    xml_text = z.read(xml_name).decode("utf-8")

root = ET.fromstring(xml_text)

objects = []
counts = Counter()

for image in root.findall("image"):
    image_id = image.attrib.get("id")
    image_name = image.attrib.get("name")
    width = int(float(image.attrib.get("width")))
    height = int(float(image.attrib.get("height")))

    for polygon in image.findall("polygon"):
        label = polygon.attrib["label"]
        points = parse_points(polygon.attrib["points"])

        obj = {
            "id": len(objects) + 1,
            "image_id": image_id,
            "image_name": image_name,
            "type": "polygon",
            "label": label,
            "points": points,
        }

        objects.append(obj)
        counts[f"polygon:{label}"] += 1

    for polyline in image.findall("polyline"):
        label = polyline.attrib["label"]
        points = parse_points(polyline.attrib["points"])

        obj = {
            "id": len(objects) + 1,
            "image_id": image_id,
            "image_name": image_name,
            "type": "polyline",
            "label": label,
            "points": points,
        }

        objects.append(obj)
        counts[f"polyline:{label}"] += 1

output = {
    "coordinate_system": {
        "origin": "top_left",
        "unit": "pixel",
        "x_axis": "right",
        "y_axis": "down",
        "perspective_correction": "not_applied"
    },
    "image": {
        "id": image_id,
        "name": image_name,
        "width": width,
        "height": height
    },
    "counts": dict(counts),
    "objects": objects
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("Saved:", OUTPUT_JSON)
print("Counts:")
for k, v in counts.items():
    print(k, v)