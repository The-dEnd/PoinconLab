import os
import json
import pandas as pd

names_dict = {
    0: "R-concentriques",
    1: "R-rayons",
    2: "R-chevrons",
    3: "R-motifs",
    4: "P-feuille",
    5: "P-triangle",
    6: "P-contour-triangle",
    7: "P-union-jack",
    8: "P-autres",
    9: "P-vertical",
    10: "Pointillés",
    11: "C-a-points",
    12: "C-a-points-blanc",
    13: "P-vertical-variation",
    14: "M-carres",
    15: "Spiderman",
    16: "autres-fig",
    17: "Arceaux",
    18: "Rouelles",
    19: "Palmettes",
    20: "Colonnettes",
    21: "Geo",
    22: "R-chevrons-variation"
}

# Fonction pour calculer l'IoU entre deux boîtes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Charger les prédictions depuis un fichier TXT
def load_predictions(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    predictions = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls, x, y, w, h, prob = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        predictions.append([x, y, w, h, int(cls), prob])
    return predictions

# Charger les labels depuis le CSV
# Charger les labels depuis le CSV avec gestion des différentes extensions d'images
def load_labels_from_csv(csv_path, image_name):
    df = pd.read_csv(csv_path, sep=";")
    
    # Essayer de trouver le bon nom d'image avec différentes extensions
    image_basename = os.path.splitext(image_name)[0]
    possible_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG"]
    
    # Chercher l'image avec les différentes extensions
    for ext in possible_extensions:
        image_name_with_ext = image_basename + ext
        labels = df[df["Picture name/number"] == image_name_with_ext]
        if not labels.empty:
            break
    else:
        labels = pd.DataFrame()  # Si aucune image correspond, retourner un DataFrame vide

    labelised_boxes = []
    die_visible_values = labels["Die visible"].tolist()

    for _, row in labels.iterrows():
        labelised_boxes.append({
            "box": {
                "x": row["x_center"],
                "y": row["y_center"],
                "w": row["width"],
                "h": row["height"]
            },
            "class": int(row["classe_id"]),
            "class_name" : names_dict[int(row["classe_id"])]
        })

    return labelised_boxes, die_visible_values


# Déterminer la catégorie et la sous-catégorie
def determine_category_and_subcategory(die_visible_values):
    if len(die_visible_values) == 1:
        category = "seul"
        subcategory = "visible" if die_visible_values[0] == 1 else "nonvisible"
    else:
        category = "plusieurs"
        if all(value == 1 for value in die_visible_values):
            subcategory = "visible"
        elif all(value == 0 for value in die_visible_values):
            subcategory = "nonvisible"
        else:
            subcategory = "deux"
    return category, subcategory

# Générer le JSON pour une image
def process_image(predictions, labels, die_visible_values, image_name, iou_threshold=0.5):
    # Regrouper les prédictions
    clusters = group_boxes_with_classes(predictions, iou_threshold)
    consolidated_results = consolidate_clusters_with_top_classes(clusters)

    # Déterminer la catégorie et sous-catégorie
    category, subcategory = determine_category_and_subcategory(die_visible_values)

    # Construire les données pour l'image
    results = {
        "global_info": {
            "num_ground_truth": len(labels),
            "num_detected": len(consolidated_results),
            "category": category,
            "subcategory": subcategory
        },
        "labelised_boxes": labels,
        "detected_boxes": []
    }

    for detected_box in consolidated_results:
        box = detected_box[:4]
        top_classes = [
            {"class": detected_box[i],"class_name" : names_dict[int(detected_box[i])], "probability": detected_box[i + 1]}
            for i in range(4, len(detected_box), 2)
        ]
        matched_class, indice_matched = match_boxes(detected_box, labels, 0.55)
        correct_class = None
        correct_class_name = None
        indice_matched_detected = None

        if matched_class is not None:
            correct_class = False
            for index, tc in enumerate(top_classes):
                if tc["class"] == matched_class:
                    correct_class = True  # Sauvegarde de l'indice où la classe correspond
                    indice_matched_detected = index
                    break  # On peut sortir de la boucle dès que la première correspondance est trouvée
            correct_class_name = names_dict[int(matched_class)]

        box_info = {
            "box": {
                "x": box[0],
                "y": box[1],
                "w": box[2],
                "h": box[3]
            },
            "top_classes": top_classes,
            "is_matched": matched_class is not None,
            "indice_matched_detected":indice_matched_detected,
            "is_matched_with": indice_matched,
            "correct_class": correct_class,
            "correct_class_name": correct_class_name
        }
        results["detected_boxes"].append(box_info)

    return {image_name: results}

# Regrouper les boîtes similaires
def group_boxes_with_classes(boxes, iou_threshold=0.5):
    clusters = []
    for box in boxes:
        added = False
        for cluster in clusters:
            if any(calculate_iou(box[:4], c[:4]) > iou_threshold for c in cluster):
                cluster.append(box)
                added = True
                break
        if not added:
            clusters.append([box])
    return clusters

# Consolider les clusters
def consolidate_clusters_with_top_classes(clusters):
    results = []
    for cluster in clusters:
        best_box = max(cluster, key=lambda b: b[5])[:4] 
        classes_probs = sorted([(b[4], b[5]) for b in cluster], key=lambda x: x[1], reverse=True)
        top_classes_probs = classes_probs[:5]
        flattened_top_classes = [item for pair in top_classes_probs for item in pair]
        results.append(list(best_box) + flattened_top_classes)
    return results

# Vérifier si une boîte correspond à une boîte labélisée
def match_boxes(detected_box, labels, iou_threshold=0.5):
    i=0
    for label in labels:
        iou = calculate_iou(detected_box[:4], label["box"].values())
        if iou > iou_threshold:
            return label["class"], i
        i+=1
    return None,None

# Traiter tous les fichiers et générer un JSON unique
def process_folder_to_single_json(pred_folder, csv_path, output_file, iou_threshold=0.5):
    final_data = {}
    for pred_file in os.listdir(pred_folder):
        if pred_file.endswith(".txt"):
            image_name = os.path.splitext(pred_file)[0] + ".jpg"
            pred_path = os.path.join(pred_folder, pred_file)

            predictions = load_predictions(pred_path)
            labels, die_visible_values = load_labels_from_csv(csv_path, image_name)
            image_data = process_image(predictions, labels, die_visible_values, image_name, iou_threshold)

            final_data.update(image_data)

    with open(output_file, "w") as file:
        json.dump(final_data, file, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Nom du modèle à traiter (dossier dans models/yolo/)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_folder = os.path.join(base_dir, "yolo", args.model_name, "runs","test", "labels")
    csv_path = os.path.join(base_dir, "data_filtered_classe_clean.csv")
    output_file = os.path.join(base_dir, "yolo", args.model_name, "resultats.json")

    process_folder_to_single_json(pred_folder, csv_path, output_file)
