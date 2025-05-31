import cv2
import torch
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
import sys

# Load YOLOv8 model
yolo_model = YOLO("yolov8_weights/yolov8n.pt")

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# Load gender classification model (assumed to be a PyTorch model)
class GenderClassifier(torch.nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

gender_model = GenderClassifier()
gender_model.load_state_dict(torch.load("gender_classifier/gender_model.pth", map_location=device))
gender_model.to(device).eval()

# Finger counting logic
def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    count = 0
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    # Thumb
    if landmarks[4].x > landmarks[2].x:
        count += 1
    return count

# Run detection and tracking
def process_video(video_path):
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]  # Get first result

        for det in results.boxes:
            cls_id = int(det.cls[0])
            xyxy = det.xyxy[0].cpu().numpy().astype(int)
            label = yolo_model.names[cls_id]
            x1, y1, x2, y2 = xyxy

            # Crop the region
            roi = frame[y1:y2, x1:x2]

            if label == 'person':
                # Gender classification
                face_img = cv2.resize(roi, (64, 64))
                face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                face_tensor = face_tensor.to(device)
                with torch.no_grad():
                    out = gender_model(face_tensor)
                    pred = torch.argmax(out, dim=1).item()
                gender = 'Man' if pred == 0 else 'Woman'
                cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            elif label == 'hand':
                # Hand landmarks
                img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        fingers = count_fingers(hand_landmarks.landmark)
                        cv2.putText(frame, f"Fingers: {fingers}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    process_video(video_path)
