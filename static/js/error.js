
// Initialisation
document.addEventListener("DOMContentLoaded", function () {
    const errorTrends = JSON.parse(document.getElementById("error-container").getAttribute("data-robert"));
    console.log("Valeur de robert récupérée :", errorTrends);


    // Variable globale pour stocker les données enrichies
    let enrichedData = [];

    // Fonction pour calculer les pourcentages et préparer les données pour le camembert
    function updatePieChart(realClass) {
        const trends = errorTrends[realClass];
        const totalErrors = Object.values(trends).reduce((sum, count) => sum + count, 0); // Somme totale des erreurs
        const labels = [];
        const data = [];
        const others = {};

        // Réinitialiser les données enrichies
        enrichedData = [];

        // Trier les classes par proportion et catégoriser
        let othersTotal = 0;
        for (const [cls, count] of Object.entries(trends)) {
            const percentage = (count / totalErrors) * 100;
            if (percentage > 10) {
                labels.push(`Classe ${cls}`);
                data.push(count);
                enrichedData.push({ label: `Classe ${cls}`, count, percentage });
            } else {
                othersTotal += count;
                others[`Classe ${cls}`] = { count, percentage };
            }
        }

        // Ajouter "Autres" si nécessaire
        if (othersTotal > 0) {
            const othersPercentage = (othersTotal / totalErrors) * 100;
            labels.push("Autres");
            data.push(othersTotal);
            enrichedData.push({ label: "Autres", count: othersTotal, percentage: othersPercentage });
        }

        // Afficher les détails des "Autres"
        const othersDetails = document.getElementById("others-details");
        const othersList = document.getElementById("others-list");
        if (othersTotal > 0) {
            othersDetails.style.display = "block";
            othersList.innerHTML = "";
            for (const [cls, { count, percentage }] of Object.entries(others)) {
                const li = document.createElement("li");
                li.textContent = `${cls}: ${count} erreurs (${percentage.toFixed(1)}%)`;
                othersList.appendChild(li);
            }
        } else {
            othersDetails.style.display = "none";
        }

        // Dessiner le camembert avec Chart.js
        const ctx = document.getElementById("error-pie-chart").getContext("2d");
        if (window.pieChart) {
            window.pieChart.destroy();
        }
        window.pieChart = new Chart(ctx, {
            type: "pie",
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#C9CBCF"
                    ],
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function (tooltipItem) {
                                const enriched = enrichedData[tooltipItem.dataIndex];
                                const count = enriched.count;
                                const percentage = enriched.percentage;
                                return `${enriched.label}: ${count} erreurs (${percentage.toFixed(1)}%)`;
                            }
                        }
                    }
                }
            }
        });
    }



    const classSelector = document.getElementById("class-selector");

    // Remplir la liste déroulante
    for (const realClass of Object.keys(errorTrends)) {
        const option = document.createElement("option");
        option.value = realClass;
        option.textContent = `Classe ${realClass}`;
        classSelector.appendChild(option);
    }

    // Afficher le camembert pour la première classe
    updatePieChart(classSelector.value);

    // Mettre à jour le camembert lors de la sélection d'une nouvelle classe
    classSelector.addEventListener("change", function () {
        updatePieChart(this.value);
    });
});
