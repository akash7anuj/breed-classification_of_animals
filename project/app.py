import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template
import os

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Breed classes (same order as training dataset)
CLASS_NAMES = ["gir", "murrah", "sahiwal", "tharparkar", "nagori"]  # apne dataset ke hisaab se update karo

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------
# LOAD MODEL
# -------------------------
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        breed = CLASS_NAMES[pred.item()]
        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    return breed, float(probs[pred.item()])

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Save file temporarily
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Predict
        breed, confidence = predict(filepath)

        return render_template(
            "index.html",
            prediction=breed,
            confidence=f"{confidence:.2f}",
            image_path=filepath
        )

    return render_template("index.html")

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
