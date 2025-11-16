// Variables globales
let currentData = null;
let currentIndex = 0;
let filteredImages = [];

// Fonction pour charger et afficher l'image et ses boîtes à un index donné
function showImage(imageIndex) {
    const imageName = filteredImages[imageIndex];
    $('#selected-image').attr('src', `/static/images/${imageName}`);

    $.getJSON(`/get_image_data/${imageName}`, function (data) {
        currentData = data; // Stocker les données de l'image sélectionnée
        const boxDetails = $('#boxes-details');
        boxDetails.empty();
    
        const canvas = $('#bounding-boxes')[0];
        const img = $('#selected-image')[0];
    
        $("#img-name").text("Image : " + imageName);
    
        // Réinitialiser la taille du canvas
        img.onload = function () {
            canvas.width = img.width;
            canvas.height = img.height;
    
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height); // Efface le canvas
    
            // Dessiner les boîtes labelisées
            data.labelised_boxes.forEach((box, index) => {
                const coords = toAbsoluteCoords(box.box, canvas.width, canvas.height);
                drawBox(ctx, coords, 'green');
    
                let labelInfo = `Labelisée #${index + 1} (${box.class_name})`;
    
                // Vérifier si une boîte détectée correspond à cette boîte
                const matchedDetected = data.detected_boxes.find(dBox => dBox.is_matched_with === index && dBox.correct_class);
            
                if (matchedDetected) {
                    labelInfo += ` - Correspondance correcte avec détectée #${data.detected_boxes.indexOf(matchedDetected) + 1}`;
                }
    
                boxDetails.append(
                    `<li>
                        <a href="#" class="toggle-box" data-index="${index}" data-type="label" style="color: green;">${labelInfo}</a>
                    </li>`
                );
            });
    
            // Dessiner les boîtes prédites
            data.detected_boxes.forEach((box, index) => {
                const coords = toAbsoluteCoords(box.box, canvas.width, canvas.height);
                drawBox(ctx, coords, 'red');
    
                let detectedInfo = `Prédite #${index + 1}`;
    
                // Ajouter les top 5 classes prédites
                const topClasses = box.top_classes.map(cls => `${cls.class_name} (${(cls.probability * 100).toFixed(1)}%)`).join(', ');
                detectedInfo += ` - Top classes : ${topClasses}`;
    
                // Vérifier si la boîte est correctement associée
                if (box.is_matched) {
                    detectedInfo += ` - Associée à labelisée #${box.is_matched_with + 1}`;
                    if (box.correct_class) {
                        detectedInfo += ` - Classe correcte hei (${box.correct_class_name})`;
                    }
                }
    
                boxDetails.append(
                    `<li>
                        <a href="#" class="toggle-box" data-index="${index}" data-type="predicted" style="color: red;">${detectedInfo}</a>
                    </li>`
                );
            });
    
            // Mettre à jour les statistiques globales
            updateGlobalStats(data);
        };
    
        // Si l'image est déjà dans le cache, forcer le redessin
        if (img.complete) {
            img.onload();
        }
    });
    
}

// Convertir les coordonnées YOLO en coordonnées absolues
function toAbsoluteCoords(box, width, height) {
    return {
        x: box.x * width - (box.w * width) / 2,
        y: box.y * height - (box.h * height) / 2,
        w: box.w * width,
        h: box.h * height,
    };
}

// Dessiner une boîte sur le canvas
function drawBox(ctx, coords, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(coords.x, coords.y, coords.w, coords.h);
}

// Fonction pour mettre à jour les statistiques globales
function updateGlobalStats(imageData) {
    const globalInfo = imageData.global_info;

    // Nombre de poinçons à détecter et détectés
    const globalText = `Nombre de poinçons à détecter : ${globalInfo.num_ground_truth}<br>
                        Nombre de poinçons détectés : ${globalInfo.num_detected}`;

    // Calcul des boîtes correspondantes
    const matchedCount = imageData.detected_boxes.filter(box => box.is_matched).length;
    const correctCount = imageData.detected_boxes.filter(box => box.correct_class === true).length;

    const matchedText = `Boîtes correctement associées (is_matched) : ${matchedCount}<br>
                         Boîtes avec classe correcte (correct_class) : ${correctCount}`;

    // Affichage des résultats dans la page
    $('#global-text').html(globalText);
    $('#matched-text').html(matchedText);
}

// Afficher/Masquer les boîtes avec redessin
$(document).on('click', '.toggle-box', function (e) {
    e.preventDefault();

    const index = $(this).data('index');
    const type = $(this).data('type');

    // Basculer l'état de masquage avec une classe
    $(this).toggleClass('hidden-box');

    // Redessiner le canvas
    redrawCanvas();
});

// Fonction de redessin du canvas
function redrawCanvas() {
    const canvas = $('#bounding-boxes')[0];
    const img = $('#selected-image')[0];
    const ctx = canvas.getContext('2d');

    if (img.complete) {
        canvas.width = img.width;
        canvas.height = img.height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Dessiner les boîtes labelisées visibles
        currentData.labelised_boxes.forEach((box, index) => {
            if (!$(`.toggle-box[data-index="${index}"][data-type="label"]`).hasClass('hidden-box')) {
                const coords = toAbsoluteCoords(box.box, canvas.width, canvas.height);
                drawBox(ctx, coords, 'green');
            }
        });

        // Dessiner les boîtes prédites visibles
        currentData.detected_boxes.forEach((box, index) => {
            if (!$(`.toggle-box[data-index="${index}"][data-type="predicted"]`).hasClass('hidden-box')) {
                const coords = toAbsoluteCoords(box.box, canvas.width, canvas.height);
                drawBox(ctx, coords, 'red');
            }
        });
    } else {
        setTimeout(redrawCanvas, 50);
    }
}

// Gestion des boutons Précédent et Suivant
$('#prev-image').on('click', function () {
    if (currentIndex > 0) {
        currentIndex--;
        showImage(currentIndex);  // Affiche l'image précédente en utilisant l'index actuel
    }
});

$('#next-image').on('click', function () {
    if (currentIndex < filteredImages.length - 1) {
        currentIndex++;
        showImage(currentIndex);  // Affiche l'image suivante en utilisant l'index actuel
    }
});



// Charger les images correspondant aux filtres
$('#apply-filters').on('click', function () {
    const category = $('#category').val();
    const subcategory = $('#subcategory').val();
    const unmatchedOnly = $('#unmatched-only').is(':checked'); // Récupérer l'état de la case à cocher

    $.ajax({
        url: '/filter_images',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ category, subcategory, unmatched_only: unmatchedOnly }),
        success: function (data) {
            let index = 0;
            filteredImages = data; // Enregistrer les images filtrées
            currentIndex = 0; // Réinitialiser l'index à la première image
            showImage(currentIndex); // Afficher la première image
            const imageList = $('#image-list');
            imageList.empty();
            data.forEach(image => {
                imageList.append(`<li class="img-link" data-image="${image}" data-index="${index}" >${image}</li>`);
                index += 1;
            });
            $("#img-dispo").text("Images disponibles ( "+data.length+"/204 )");
            // Lorsqu'un élément est cliqué, met à jour l'image et l'index courant
            $(".img-link").on('click', function () {
                currentIndex = $(this).data('index');
                showImage(currentIndex);
            });
        },
        error: function (xhr, status, error) {
            console.error("Erreur:", status, error);
        }
    });
});

