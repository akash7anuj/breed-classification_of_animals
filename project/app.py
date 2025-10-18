import os
import time
import torch
import torch.nn as nn
from PIL import Image
import cv2
from torchvision import transforms, models
from ultralytics import YOLO
from flask import Flask, request, render_template

# ------------------------- CONFIG -------------------------
MODEL_PATH = r"C:\Users\DELL\Desktop\breed-classification_of_animals\project\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss',
    'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian',
    'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh',
    'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri',
    'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi',
    'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------- LOAD MODELS -------------------------
def load_breed_model():
    """Load trained ResNet50 model for cattle breed classification."""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

breed_model = load_breed_model()
yolo_model = YOLO('yolov8m.pt')

# ------------------------- FLASK APP -------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

def clear_static_folder():
    """Remove all previously saved images before saving a new one."""
    for f in os.listdir(STATIC_DIR):
        try:
            os.remove(os.path.join(STATIC_DIR, f))
        except Exception:
            pass


def classify_breed(crop_img):
    """Classify cattle breed using the ResNet50 model."""
    image = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = breed_model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        breed_name = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()

    return breed_name, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        threshold = float(request.form.get("confidence", 40)) / 100.0

        if not file or file.filename == "":
            return render_template("index.html", error="⚠️ Please select an image first.")

        clear_static_folder()  # 👈 Delete all previous images

        # Save file
        unique_name = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(STATIC_DIR, unique_name)
        file.save(filepath)

        # Load image
        img = cv2.imread(filepath)
        results = yolo_model(filepath)
        breed_list = []
        other_objects = []   # 👈 New list
        found_cattle = False

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = yolo_model.model.names[cls].lower()
            conf = float(box.conf[0])

            if conf < threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            color = (0, 0, 255)
            display_label = f"{label} ({conf*100:.1f}%)"

            if any(word in label for word in ["cow", "buffalo", "cattle", "ox", "animal"]):
                found_cattle = True
                color = (0, 255, 0)
                crop_img = img[y1:y2, x1:x2]

                if crop_img.size != 0:
                    breed, breed_conf = classify_breed(crop_img)
                    display_label = f"{breed} ({breed_conf*100:.1f}%)"
                    breed_list.append(display_label)
            else:
                # 👇 Store non-cattle detections
                other_objects.append(f"{label} ({conf*100:.1f}%)")

            # Draw boxes
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img, display_label, (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save output image
        output_path = os.path.join(STATIC_DIR, f"detected_{unique_name}")
        cv2.imwrite(output_path, img)
        os.remove(filepath)

        if not found_cattle:
            return render_template(
                "index.html",
                error="❌ No cow or buffalo detected above confidence threshold!",
                image_path=f"detected_{unique_name}",
                confidence=int(threshold * 100),
                other_objects=other_objects  # 👈 Pass even if no cattle
            )

        return render_template(
            "index.html",
            prediction=f"✅ {len(breed_list)} cattle detected and classified!",
            breed_list=breed_list,
            other_objects=other_objects,  # 👈 Added here
            image_path=f"detected_{unique_name}",
            confidence=int(threshold * 100)
        )        

    return render_template("index.html", confidence=40)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
