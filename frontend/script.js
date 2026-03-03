const previewImage = document.getElementById("previewImage");
const resultBox = document.getElementById("resultBox");
const predictionText = document.getElementById("predictionText");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");
const loading = document.getElementById("loading");

document.getElementById("imageInput").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file) {
        previewImage.src = URL.createObjectURL(file);
        previewImage.style.display = "block";
    }
});

async function predict() {
    const fileInput = document.getElementById("imageInput");

    if (!fileInput.files[0]) {
        alert("Please upload an image first.");
        return;
    }

    loading.style.display = "block";
    resultBox.style.display = "none";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loading.style.display = "none";
        resultBox.style.display = "block";

        predictionText.innerText = data.prediction;

        confidenceFill.style.width = data.confidence + "%";
        confidenceText.innerText = data.confidence + "% confidence";

        if (data.prediction === "PNEUMONIA") {
            confidenceFill.style.background = "linear-gradient(90deg, #ef4444, #dc2626)";
            predictionText.style.color = "#ef4444";
        } else {
            confidenceFill.style.background = "linear-gradient(90deg, #22c55e, #16a34a)";
            predictionText.style.color = "#22c55e";
        }

    } catch (error) {
        loading.style.display = "none";
        alert("Backend connection failed.");
        console.error(error);
    }
}
