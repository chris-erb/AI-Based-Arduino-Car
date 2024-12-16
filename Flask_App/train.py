from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data=r"C:\Users\chris\PycharmProjects\ArduCar\dataset\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device='cpu'
)
