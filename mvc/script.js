let video = document.createElement("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let toggleButton = document.getElementById("toggle-camera");
let button = document.getElementById(".corner-button");
let mediaContainer = document.querySelector(".media-container");
let streaming = false;
let lastFaces = [];
let frameId = -1;

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.play();
            streaming = true;
            toggleButton.textContent = "Dừng Camera";
            mediaContainer.classList.add("camera-on");
            mediaContainer.classList.remove("image-on");
            processFrame();
        })
        .catch(err => console.error("Lỗi mở camera:", err));
}

function stopCamera() {
    streaming = false;
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    toggleButton.textContent = "Bật Camera";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    mediaContainer.classList.remove("camera-on");
}

toggleButton.addEventListener("click", () => {
    if (streaming) {
        stopCamera();
    } else {
        startCamera();
    }
});

async function processFrame() {
    if (frameId === 3){
        frameId = 0;
    } else {
        frameId++;
    }
    console.log("Frame ID:", frameId);
    if (!streaming) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    drawBoundingBoxes(lastFaces);
    let imageBase64 = canvas.toDataURL("image/jpeg").split(",")[1];
    let jsonData = JSON.stringify({ "image": imageBase64, "frameId": frameId });

    try {
        let response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: jsonData
        });

        let data = await response.json();
        if (data.faces) lastFaces = data.faces;
    } catch (err) {
        console.error("Lỗi gửi frame đến Flask:", err);
    }
    setTimeout(processFrame, 100);
}

function drawBoundingBoxes(faces) {
    faces.forEach(face => {
        let { x1, y1, x2, y2 } = face.bounding_box;
        let emotion = face.emotion;
        let probability = (face.probability).toFixed(2) + "%";
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = "lime";
        ctx.font = "18px Arial";
        ctx.fillText(`${emotion} (${probability})`, x1 + 5, y1 - 5);
    });
}



