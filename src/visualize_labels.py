import random
from pathlib import Path
import cv2
import yaml


random.seed(42)
dataset_path = Path("data")
split = "valid"
images_path = dataset_path / split / "images"
labels_path = dataset_path / split / "labels"
output_path = Path("runs/dataset_preview")
with (dataset_path / "data.yaml").open("r", encoding="utf-8") as file:
    dataset_config = yaml.safe_load(file)

class_names = dataset_config["names"]
image_files = [
    path
    for path in images_path.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
]
image_path = random.choice(image_files)
label_path = labels_path / f"{image_path.stem}.txt"
image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Could not read image: {image_path}")
image_height, image_width = image.shape[:2]
with label_path.open("r", encoding="utf-8") as file:
    for line in file:
        class_id, x_center, y_center, box_width, box_height = map(
            float, line.split()
        )
        class_id = int(class_id)
        x1 = int((x_center - box_width / 2) * image_width)
        y1 = int((y_center - box_height / 2) * image_height)
        x2 = int((x_center + box_width / 2) * image_width)
        y2 = int((y_center + box_height / 2) * image_height)
        class_name = class_names[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
output_path.mkdir(parents=True, exist_ok=True)
saved_image = output_path / f"{split}_annotation_preview.jpg"
cv2.imwrite(str(saved_image), image)
print(f"Original image: {image_path}")
print(f"Label file: {label_path}")
print(f"Preview saved to: {saved_image}")