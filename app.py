from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from collections import defaultdict
import json
import numpy as np
import os
import math
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from werkzeug.utils import secure_filename
import zipfile
import subprocess
import sys
from pathlib import PosixPath
import torch
from flask import send_file, abort

app = Flask(__name__)
app.secret_key = 'ta_clé_secrète'  # Pour les messages flash

# Charger le fichier JSON
with open("resultats.json", "r") as f:
    data = json.load(f)
#################################################### MEHDI ####################################################
def charger_resultats():
    return data

def calculer_statistiques(data):
    total_images = len(data)
    total_ground_truth = 0
    total_detected = 0
    total_matched = 0
    total_correct_class = 0
    total_detected_boxes = 0
    total_first_correct_class = 0

    for image_data in data.values():
        global_info = image_data["global_info"]
        total_ground_truth += global_info["num_ground_truth"]
        total_detected += global_info["num_detected"]

        for box in image_data["detected_boxes"]:
            total_detected_boxes += 1
            if box["is_matched"]:
                total_matched += 1
                if box["correct_class"]:
                    total_correct_class += 1
                    if box["indice_matched_detected"] == 0:
                        total_first_correct_class+=1

    taux_detection_par_image = (total_matched / total_ground_truth) * 100 if total_ground_truth > 0 else 0
    taux_global_association = (total_matched / total_detected_boxes) * 100 if total_detected_boxes > 0 else 0
    taux_global_classification_correcte = (total_correct_class / total_matched) * 100 if total_matched > 0 else 0
    score_combine = math.sqrt(taux_detection_par_image/100 * taux_global_association/100 * taux_global_classification_correcte/100)*100
    taux_first_correct_class = (total_first_correct_class / total_correct_class) * 100 if total_correct_class > 0 else 0

    return {
        "total_images": total_images,
        "taux_detection_par_image": round(taux_detection_par_image, 1),
        "taux_detection_par_image_text": f"{total_matched}/{total_ground_truth}",
        "taux_global_association": round(taux_global_association, 1),
        "taux_global_association_text":f"{total_matched}/{total_detected_boxes}",
        "taux_global_classification_correcte": round(taux_global_classification_correcte, 1),
        "taux_global_classification_correcte_text":f"{total_correct_class}/{total_matched}",
        "score_combine": round(score_combine, 1),
        "taux_first_correct_class": round(taux_first_correct_class, 1),
        "taux_first_correct_class_text":f"{total_first_correct_class}/{total_correct_class}"
    }

def calculer_statistiques_par_categorie(data):
    categories = ['seul', 'plusieurs']
    subcategories = ['visible', 'nonvisible', 'deux']

    statistiques_par_categorie = {}
    
    for category in categories:
        for subcategory in subcategories:
            if category == 'seul' and subcategory == 'deux':
                continue
            
            total_images = 0
            category_total_ground_truth = 0
            category_total_detected_boxes = 0
            category_total_matched = 0
            category_total_correct_class = 0
            
            for image_data in data.values():
                global_info = image_data["global_info"]
                
                if global_info["category"] == category and global_info["subcategory"] == subcategory:
                    total_images += 1
                    category_total_ground_truth += global_info["num_ground_truth"]
                    
                    for box in image_data["detected_boxes"]:
                        category_total_detected_boxes += 1
                        if box["is_matched"]:
                            category_total_matched += 1
                            if box["correct_class"]:
                                category_total_correct_class += 1
            
            taux_detection_par_image = (category_total_matched / category_total_ground_truth) * 100 if category_total_ground_truth > 0 else 0
            taux_global_association = (category_total_matched / category_total_detected_boxes) * 100 if category_total_detected_boxes > 0 else 0
            taux_global_classification_correcte = (category_total_correct_class / category_total_matched) * 100 if category_total_matched > 0 else 0
            score_combine = (taux_detection_par_image + taux_global_association + taux_global_classification_correcte) / 3

            statistiques_par_categorie[(category, subcategory)] = {
                "taux_detection_par_image": round(taux_detection_par_image, 1),
                "taux_global_association": round(taux_global_association, 1),
                "taux_global_classification_correcte": round(taux_global_classification_correcte, 1),
                "score_combine": round(score_combine, 1)
            }
    
    return statistiques_par_categorie

def calculer_matrice_confusion(class_names):
    matrix_confusion = {class_name: defaultdict(int) for class_name in class_names}

    # Parcours des images
    for image_data in data.values():  # images_data est ton dictionnaire contenant les infos des images
        for detected_box in image_data["detected_boxes"]:
            if detected_box["is_matched"]:
                correct_class_name = detected_box["correct_class_name"]
                if detected_box["correct_class"]:
                    # Prédiction correcte
                    matrix_confusion[correct_class_name][correct_class_name] += 1
                else:
                    # Mauvaise prédiction
                    detected_class_name = detected_box["top_classes"][0]["class_name"]
                    matrix_confusion[correct_class_name][detected_class_name] += 1
    return matrix_confusion
def convertir_matrice_dict_vers_numpy(confusion_dict, class_names):
    n = len(class_names)
    class_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    confusion_matrix = np.zeros((n, n), dtype=int)

    for true_class, preds in confusion_dict.items():
        true_idx = class_index[true_class]
        for pred_class, count in preds.items():
            pred_idx = class_index[pred_class]
            confusion_matrix[true_idx, pred_idx] = count
    return confusion_matrix

def calculer_f1_pondere_par_classe(confusion_matrix, class_names):
    n = len(class_names)
    total_support = 0
    weighted_sum_f1 = 0

    for i in range(n):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        weighted_sum_f1 += f1 * support
        total_support += support

    f1_pondere = weighted_sum_f1 / total_support if total_support > 0 else 0
    return round(f1_pondere * 100, 1)
                
@app.route('/analyse')
def analyse():
    data = charger_resultats()
    statistiques = calculer_statistiques(data)
    statistiques_par_categorie = calculer_statistiques_par_categorie(data)

    # Calcul de la matrice de confusion et des stats par classe ici même A CORRIGE 
    class_names = obtenir_classes(data)
    confusion_dict = calculer_matrice_confusion(class_names)

    error_trends = calculate_error_trends()
    extract_error_predictions()
    confusion_matrix = convertir_matrice_dict_vers_numpy(confusion_dict, class_names)
    stats = calculer_stats_par_classe(confusion_matrix)
    f1_pondere = calculer_f1_pondere_par_classe(confusion_matrix, class_names)
        # Génération de la matrice de confusion visuelle
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, 
                    annot_kws={"size": 10})

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.xlabel("Prédictions", fontsize=12)
    plt.ylabel("Classes Réelles", fontsize=12)
    plt.title("Matrice de Confusion", fontsize=14)

    plt.tight_layout()  # Ajuste les marges


    # Sauvegarde de la figure dans un buffer mémoire
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()



    


    return render_template(
    'analyse.html', 
    statistiques=statistiques, 
    statistiques_par_categorie=statistiques_par_categorie,
    images=list(data.keys()),
    title="Plateforme d'Analyse des Poinçons",
    confusion_matrix=confusion_matrix.tolist(),
    class_names=class_names,
    error_trends=error_trends,
    stats=stats,
    f1_pondere=f1_pondere,
    image_base64=image_base64,  # Ajout de l'image encodée
)

 #################################################### MEHDI ####################################################
# Route principale
@app.route("/")
def index():
    # Extraire toutes les catégories et sous-catégories uniques
    categories = set()
    subcategories = set()
    for image_info in data.values():
        categories.add(image_info["global_info"]["category"])
        subcategories.add(image_info["global_info"]["subcategory"])

    return render_template(
        "index.html",
        categories=list(categories),
        subcategories=list(subcategories),
    )
#################################################### MEHDI MAT CONFUSION ####################################################
def obtenir_classes(data):
    """ Récupère toutes les classes mentionnées dans les ground truth et détections. """
    classes = set()
    for image_data in data.values():
        # Récupérer les classes ground truth
        for lb in image_data["labelised_boxes"]:
            classes.add((lb["class"], lb["class_name"]))
        # Récupérer les classes détectées
        for db in image_data["detected_boxes"]:
            for top_class in db["top_classes"]:
                classes.add((top_class["class"], top_class["class_name"]))
    return [c[1] for c in sorted(list(classes), key=lambda x: x[0])]

def convertir_matrice_dict_vers_numpy(confusion_dict, class_names):
    n = len(class_names)
    class_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    confusion_matrix = np.zeros((n, n), dtype=int)

    for true_class, preds in confusion_dict.items():
        true_idx = class_index[true_class]
        for pred_class, count in preds.items():
            pred_idx = class_index[pred_class]
            confusion_matrix[true_idx, pred_idx] = count
    return confusion_matrix


def calculer_stats_par_classe(confusion_matrix):
    stats = []
    n = confusion_matrix.shape[0]
    for i in range(n):
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, :].sum() - tp
        fp = confusion_matrix[:, i].sum() - tp
        tn = confusion_matrix.sum() - (tp + fn + fp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        stats.append({
            "class_idx": i,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        })
    return stats
# #################################################### MEHDI ####################################################


def calculate_error_trends():

    # Initialiser la structure des tendances d'erreurs
    error_trends = defaultdict(lambda: defaultdict(int))

    # Parcourir chaque image
    mama = 0
    for image_name, image_data in data.items():
        labelised_boxes = image_data.get("labelised_boxes", [])
        detected_boxes = image_data.get("detected_boxes", [])

        # Pour chaque boîte détectée
        for detected_box in detected_boxes:
            if detected_box["is_matched"] and not detected_box["correct_class"]:
         
                real_class = detected_box["correct_class_name"]
             
                for top_class in detected_box["top_classes"]:
                    predicted_class = top_class["class_name"]

                    error_trends[real_class][predicted_class] += 1

    # Convertir en un dictionnaire classique pour le rendre JSON-compatible
    error_trends = {real_class: dict(predicted_classes) for real_class, predicted_classes in error_trends.items()}
    return error_trends


def extract_error_predictions(output_file="error_predictions2.json"):
    # Initialiser la structure pour stocker les prédictions par classe réelle
    error_predictions = defaultdict(list)

    # Parcourir chaque image
    for image_name, image_data in data.items():
        detected_boxes = image_data.get("detected_boxes", [])

        # Pour chaque boîte détectée
        for detected_box in detected_boxes:
            if detected_box["is_matched"]:  # On s'intéresse aux boîtes appariées uniquement
                real_class = detected_box["correct_class_name"]

                # Extraire les classes prédites (top 5)
                predicted_classes = {top_class["class_name"] for top_class in detected_box["top_classes"]}
               
                
                # Ajouter cet ensemble au dictionnaire
                error_predictions[real_class].append(predicted_classes)
    # Conversion des ensembles en listes
    def serialize_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError("Type non sérialisable : {}".format(type(obj)))

    # Sauvegarde en JSON
    with open('error_predictions2.json', 'w', encoding='utf-8') as json_file:
        json.dump(error_predictions, json_file, default=serialize_sets, indent=4, ensure_ascii=False)

        print(f"Les prédictions d'erreurs ont été enregistrées dans {output_file}.")


@app.route("/filter_images", methods=["POST"])
def filter_images():
    filters = request.get_json()  # Récupérer les données JSON envoyées
    category_filter = filters.get("category")
    subcategory_filter = filters.get("subcategory")
    unmatched_only = filters.get("unmatched_only", False)

    # Filtrer les images
    filtered_images = []
    for image_name, image_info in data.items():
        category = image_info["global_info"]["category"]
        subcategory = image_info["global_info"]["subcategory"]

        # Vérifier si aucune boîte détectée n'a `is_matched = True`
        if unmatched_only:
            has_correct_match = any(box.get("is_matched", False) for box in image_info["detected_boxes"])
            if has_correct_match:
                continue  # Ignorer les images qui ont des correspondances correctes

        # Appliquer les autres filtres
        if (category_filter in [category, "toutes"]) and (subcategory_filter in [subcategory, "toutes"]):
            filtered_images.append(image_name)

    return jsonify(filtered_images)


@app.route("/get_image_data/<image_name>")
def get_image_data(image_name):
    image_data = data.get(image_name, {})
    return jsonify(image_data)











UPLOAD_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_model', methods=['GET'])
def upload_model_form():
    return render_template('upload_model.html')

@app.route('/upload_zip', methods=['POST'])
def upload_zip():
    model_type = request.form.get('model_type')
    model_name = request.form.get('model_name')
    zip_file = request.files['model_file']

    try:
        model_dir = os.path.join('models', model_type, model_name)
        os.makedirs(model_dir, exist_ok=True)

        zip_path = os.path.join(model_dir, 'model.zip')
        zip_file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        os.remove(zip_path)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/run_inference/<model_name>', methods=['POST'])
def run_inference_route(model_name):
    try:
        subprocess.run(['python', 'models/run_inference.py', model_name], check=True)
        return jsonify({'status': 'ok'})
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'message': 'Erreur inférence : ' + str(e)})

@app.route('/generate_json/<model_name>', methods=['POST'])
def generate_json(model_name):
    try:
        subprocess.run(['python', 'models/traitement_json.py', model_name], check=True)
        return jsonify({'status': 'ok'})
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'message': 'Erreur JSON : ' + str(e)})



@app.route('/inference_log/<model_name>')
def inference_log(model_name):
    try:
        log_path = os.path.join('models', 'yolo', model_name, 'inference_log.txt')
        with open(log_path, 'r') as f:
            content = f.read()
        return jsonify({'log': content})
    except Exception as e:
        return jsonify({'log': '', 'error': str(e)})
    
@app.route('/check_model_exists/<model_type>/<model_name>')
def check_model_exists(model_type, model_name):
    path = os.path.join('models', model_type, model_name)
    exists = os.path.exists(path)
    return jsonify({'exists': exists})


@app.route('/result_image/<session>/<model>/<filename>')
def serve_result_image(session, model, filename):
    # Recomposer le chemin
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PRED_DIR = os.path.join(BASE_DIR, 'models', 'prediction', 'outputs')
    img_path = os.path.join(PRED_DIR, session, model, filename)

    # Optionnel : sécurité basique
    if not os.path.isfile(img_path):
        return abort(404)

    return send_file(img_path)

@app.route('/predict', methods=['GET'])
def predict_form():
    # trouver tous les modèles dispo dans yolo
    yolo_models_dir = os.path.join("models", "yolo")
    models = []
    for name in os.listdir(yolo_models_dir):
        weight_path = os.path.join(yolo_models_dir, name, "runs", "train", "exp", "weights", "best.pt")
        if os.path.exists(weight_path):
            models.append((name, weight_path))
    return render_template("predict.html", models=models)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    image = request.files['image']
    selected_models = request.form.getlist('models')

    save_path = os.path.join("static", "uploads")
    os.makedirs(save_path, exist_ok=True)
    img_path = os.path.join(save_path, image.filename)
    image.save(img_path)

    # convertir model selectionnés en tuples (name, path)
    models = [tuple(m.split("::")) for m in selected_models]

    # lancer la prédiction
    from models.prediction.predict_image import predict_on_image
    results = predict_on_image(img_path, models)


    return render_template("predict.html", models=models, results=results)





# Lancer l'application
if __name__ == "__main__":
    app.run(debug=False)
