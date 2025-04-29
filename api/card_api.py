from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import os
from flask_cors import CORS

# Define model class again (matches training script)
class SimpleCardClassifier(torch.nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=False)
        self.features = torch.nn.Sequential(*list(self.base_model.children())[:-1])
        self.classifier = torch.nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Flask setup
app = Flask(__name__)
CORS(app, origins=["https://joesinnott.github.io"])  # Allow frontend URL

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
model = SimpleCardClassifier()
model.load_state_dict(torch.load("api/model_weights.pth", map_location=device, weights_only=True))  # path to trained weights
model.to(device)
model.eval()

# Label mapping
target_to_class = {0: 'ace of clubs', 
                   1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades', 4: 'eight of clubs', 5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades', 8: 'five of clubs', 9: 'five of diamonds', 10: 'five of hearts', 11: 'five of spades', 12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 15: 'four of spades', 16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades', 20: 'joker', 21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts', 24: 'king of spades', 25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts', 28: 'nine of spades', 29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts', 32: 'queen of spades', 33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts', 36: 'seven of spades', 37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts', 40: 'six of spades', 41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts', 44: 'ten of spades', 45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts', 48: 'three of spades', 49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts', 52: 'two of spades'}

# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        image = Image.open(file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, dim=1)
            pred_idx = pred_idx.item()
            pred_class = target_to_class[pred_idx]

        return jsonify({"prediction": pred_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API
if __name__ == "__main__":
    app.run()
