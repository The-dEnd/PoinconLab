import os
import subprocess
import sys
import time

# Paramètres
MODEL_TYPE = "yolo"  # pour l'instant seulement YOLO est pris en charge
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_FOLDER = os.path.join(BASE_DIR, "yolov5")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def run_yolo_inference(model_name):
    print("[1/1] Lancement de l'inférence YOLO...")

    weights_path = os.path.join(BASE_DIR, MODEL_TYPE, model_name, "runs", "train", "exp", "weights", "best.pt")
    save_dir = os.path.join(BASE_DIR, MODEL_TYPE, model_name, "runs", "test")
    log_path = os.path.join(BASE_DIR, MODEL_TYPE, model_name, "inference_log.txt")

    command = [
        sys.executable, os.path.join(YOLO_FOLDER, "val.py"),
        "--weights", weights_path,
        "--data", CONFIG_PATH,
        "--task", "test",
        "--save-txt",
        "--name", ".",
        "--project", save_dir,
        "--save-conf",
    ]

    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        buffer = []
        last_write_time = time.time()

        for line in process.stdout:
            buffer.append(line)
            if time.time() - last_write_time >= 45:
                log_file.writelines(buffer)
                log_file.flush()
                buffer = []
                last_write_time = time.time()

        if buffer:
            log_file.writelines(buffer)
            log_file.flush()
        process.wait()

    print("[1/1] Inférence terminée.")


def main(model_name):
    try:
        run_yolo_inference(model_name)
        print("Inférence réussie ✅")
    except Exception as e:
        print("Erreur pendant l'inférence :", str(e))
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Nom du modèle à traiter")
    args = parser.parse_args()
    main(args.model_name)