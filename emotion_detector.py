import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# Model definition
# ----------------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Load model
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.path.dirname(__file__), "best_emotion_cnn.pth")

model = EmotionCNN(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

EMOTIONS = ["angry", "happy", "sad", "stressed"]

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"emotion": "error", "message": "No file uploaded."})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"emotion": "error", "message": "No file selected."})

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).convert("L")
        img = transform(img).unsqueeze(0).to(DEVICE)

        # Model prediction
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            predicted_emotion = EMOTIONS[pred.item()]

        image_url = f"/{app.config['UPLOAD_FOLDER']}/{filename}"
        return jsonify({"emotion": predicted_emotion, "image_url": image_url})

    except Exception as e:
        return jsonify({"emotion": "error", "message": str(e)})

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
