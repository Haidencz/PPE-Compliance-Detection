from collections import Counter
from pathlib import Path
import yaml

dataset_path = Path("data")
yaml_path = dataset_path / "data.yaml"
labels_path = dataset_path / "train" / "labels"
with yaml_path.open("r", encoding="utf-8") as file:
    dataset_config = yaml.safe_load(file)
class_names = dataset_config["names"]
class_counts = Counter()

label_files = list(labels_path.glob("*.txt"))
for label_file in label_files:
    with label_file.open("r", encoding="utf-8") as file:
        for line in file:
            values = line.split()
            if not values:
                continue
            class_id = int(values[0])
            class_counts[class_id] += 1
print(f"Training label files: {len(label_files)}")
print("\nObject distribution:")
for class_id, class_name in enumerate(class_names):
    count = class_counts[class_id]
    print(f"{class_id}: {class_name:<15} {count}")