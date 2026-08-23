from pathlib import Path
from ultralytics import YOLO


image_path = Path("sample_images/test.jpg")

if not image_path.exists():
    raise FileNotFoundError(f"Test image not found: {image_path}")

# Load a small model pretrained on general objects
model = YOLO("yolo11n.pt")

# Run object detection and save the labelled image
results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True
)

# Print each detected object
for box in results[0].boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    class_name = model.names[class_id]

    print(f"{class_name}: {confidence:.2f}")