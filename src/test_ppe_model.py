from pathlib import Path

from ultralytics import YOLO


model_path = Path("models/ppe_best.pt")
image_path = Path("sample_images/test.jpg")
output_path = Path("runs/ppe_prediction").resolve()

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

model = YOLO(str(model_path))

results = model.predict(
    source=str(image_path),
    conf=0.25,
    save=True,
    project=str(output_path.parent),
    name=output_path.name,
    exist_ok=True
)

for box in results[0].boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    class_name = model.names[class_id]

    print(f"{class_name}: {confidence:.2f}")