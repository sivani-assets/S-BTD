function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
}

function predict() {
    let fileInput = document.getElementById("fileUpload");
    if (fileInput.files.length === 0) {
        alert("Please upload an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("output").style.display = "block";
        document.getElementById("result").textContent = data.result;
        document.getElementById("confidence").textContent = (data.confidence * 100).toFixed(2) + "%";
        document.getElementById("gradcamImage").src = data.gradcam;
        document.getElementById("featureMapImage").src = data.feature_map;
    })
    .catch(error => console.error("Error:", error));
}

function downloadPDF() {
    let reportData = {
        name: "John Doe",
        age: "45",
        id: "12345",
        result: document.getElementById("result").textContent,
        confidence: document.getElementById("confidence").textContent,
        gradcam: document.getElementById("gradcamImage").src
    };

    fetch("/download_pdf", {
        method: "POST",
        body: JSON.stringify(reportData),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.blob())
    .then(blob => {
        let link = document.createElement("a");
        link.href = window.URL.createObjectURL(blob);
        link.download = "Patient_Report.pdf";
        link.click();
    })
    .catch(error => console.error("Error:", error));
}
