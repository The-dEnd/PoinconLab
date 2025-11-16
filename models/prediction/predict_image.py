import os
import sys
import yaml
import time
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont

CONFIDENCE_THRESHOLD = 0.3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE_DIR, "outputs")
DETECT_SCRIPT = os.path.join(BASE_DIR, "..", "yolov5", "detect.py")
CLASS_YAML = os.path.join(BASE_DIR, "class_names.yaml")

# Charger les noms de classes
with open(CLASS_YAML, 'r', encoding='utf-8') as f:
    names_dict = yaml.safe_load(f)['names']

def run_detection(model_path, image_path, output_folder):
    command = [
        sys.executable, DETECT_SCRIPT,
        '--weights', model_path,
        '--source', image_path,
        '--conf-thres', '0.001',  # on filtre après
        '--save-txt',
        '--save-conf',
        '--project', output_folder,
        '--name', '.',
        '--exist-ok'
    ]
    subprocess.run(command, check=True)

def draw_boxes(image_path, txt_path, out_path, threshold):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # police pour les labels
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    print(txt_path)    

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id, x, y, w, h, conf = int(parts[0]), *map(float, parts[1:])
            if conf < threshold:
                continue
            name = names_dict.get(cls_id, str(cls_id))
            img_w, img_h = image.size
            x1 = int((x - w / 2) * img_w)
            y1 = int((y - h / 2) * img_h)
            x2 = int((x + w / 2) * img_w)
            y2 = int((y + h / 2) * img_h)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 10), f"{name} ({conf:.2f})", fill="red", font=font)

    image.save(out_path)

def predict_on_image(image_path, model_list):
    timestamp = str(int(time.time()))
    session_name = f"session_{timestamp}"
    session_dir = os.path.join(PRED_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)

    result_images = []
    for model_name, model_path in model_list:
        model_dir = os.path.join(session_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        run_detection(model_path, image_path, model_dir)

        result_img_path = os.path.join(model_dir, os.path.basename(image_path))

        # Générer une URL vers la route Flask
        image_url = f"/result_image/{session_name}/{model_name}/{os.path.basename(image_path)}"
        result_images.append((model_name, image_url))

    return result_images


if __name__ == '__main__':
    # Pour test local : python predict_image.py test.jpg model1 path/to/pt1.pt model2 path/to/pt2.pt
    img = sys.argv[1]
    args = sys.argv[2:]
    models = list(zip(args[::2], args[1::2]))
    results = predict_on_image(img, models)
    print("Résultats générés :")
    for name, path in results:
        print(f"{name} : {path}")
