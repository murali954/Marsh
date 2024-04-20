from ultralytics import YOLO
import cv2
import numpy as np

# Load a model
model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('p.mp4')
count = 0

while True:
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    img = cv2.resize(img, (640, 320))  # Resize to match model's input size
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results.pred:
        boxes = r[:, :4]  # Boxes object for bbox outputs
        scores = r[:, 4]  # Confidence scores
        classes = r[:, 5]  # Class labels

        for box, score, cls in zip(boxes, scores, classes):
            x, y, x1, y1 = box.astype(int)
            c = class_names[int(cls)]
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
            cv2.putText(img, f"{c}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
