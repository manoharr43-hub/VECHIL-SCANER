from ultralytics import YOLO

# Load pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Train on custom dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="vehicle_ai_model"
)

print("Training Completed 🚀")
