from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

test_image_path = "dataset/train/images/curve1_jpg.rf.7dc76883e405a622201f74646cd9e676.jpg"

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(test_image_path, conf=0.2, show=False)
print("\nPrediction Details:")
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    confidence = float(box.conf)
    class_name = model.names[int(box.cls)]

    print(f"Bounding Box: [{x1}, {y1}, {x2}, {y2}], Confidence: {confidence:.2f}, Class: {class_name}")

frame = cv2.imread(test_image_path)
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    confidence = float(box.conf)
    class_name = model.names[int(box.cls)]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(frame_rgb)
plt.axis("off")
plt.show()
