import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from flask_socketio import SocketIO
import os

last_emotion_results = {}
last_emotion = ""
last_probs_percent = {}
last_prediction = ""
# Khởi tạo Flask
app = Flask(__name__)
CORS(app)
SocketIO(app)

# Load mô hình YOLOv11 để nhận diện khuôn mặt
face_model = YOLO("assets/model/yolov11n-face.pt")

# Load mô hình nhận diện cảm xúc
class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.fc1 = nn.Linear(512 * 3 * 3, 256)
        self.batch_norm_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.batch_norm_fc1(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.batch_norm_fc2(x)
        x = self.dropout_fc2(x)

        x = self.fc3(x)
        return x

emotion_model = ConvNet(num_classes=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model.load_state_dict(torch.load('assets/model/model69_adam.pth', map_location=device))
emotion_model.eval()

# Tiền xử lý ảnh khuôn mặt
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Nhãn cảm xúc
emotion_labels = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

@app.route("/predict", methods=["POST"])
def predict():
    global last_emotion_results, last_emotion, last_probs_percent, last_prediction
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Giải mã ảnh từ Base64
        image_data = base64.b64decode(data["image"])
        image_id = data["frameId"]
        image = Image.open(BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)

        # Nhận diện khuôn mặt bằng YOLOv11
        results = face_model(img_array)

        faces_detected = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Cắt vùng khuôn mặt
                face_crop = img_array[y1:y2, x1:x2]
                face_crop_pil = Image.fromarray(face_crop).convert("L")
                face_tensor = transform(face_crop_pil).unsqueeze(0)
                if image_id == 0:
                # Dự đoán cảm xúc
                    with torch.no_grad():
                        output = emotion_model(face_tensor)
                        probabilities = torch.softmax(output, dim=1)[0]
                        prediction = torch.argmax(probabilities).item()
                        
                    probs_percent = {emotion_labels[i]: round(probabilities[i].item() * 100, 2) for i in range(6)}
                    last_prediction = prediction
                
                
                last_emotion = emotion_labels[last_prediction]
                last_emotion_results = probs_percent[last_emotion]
                last_probs_percent = probs_percent
                
                detected_face = {
                    "emotion": last_emotion,
                    "probability": last_emotion_results,
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "all_probabilities": last_probs_percent,
                    "face_confidence": confidence
                }
                faces_detected.append(detected_face)

        return jsonify({"faces": faces_detected})

    except Exception as e:
        print("LỖI SERVER:", str(e))  # In lỗi ra terminal Flask
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT, debug=False)
