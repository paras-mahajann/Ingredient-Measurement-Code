function uploadImage() {
    let fileInput = document.getElementById("upload");
    let file = fileInput.files[0];

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", { method: "POST", body: formData })
    .then(response => response.json())
    .then(data => displayResults(data))
    .catch(error => console.error("Error:", error));
}

function startWebcam() {
    let video = document.getElementById("webcam");
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; video.style.display = "block"; })
        .catch(error => console.error("Webcam error:", error));
}

function captureImage() {
    let video = document.getElementById("webcam");
    let canvas = document.getElementById("captureCanvas");
    let context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        let formData = new FormData();
        formData.append("file", blob, "webcam.jpg");

        fetch("/predict", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => displayResults(data))
        .catch(error => console.error("Error:", error));
    });
}

function displayResults(data) {
    document.getElementById("material").innerText = data.material;
    document.getElementById("container").innerText = data.container;
    document.getElementById("weight").innerText = data.estimated_weight;

    let resultDiv = document.getElementById("result");
    resultDiv.style.display = "block";
}
