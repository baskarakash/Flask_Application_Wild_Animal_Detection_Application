function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {  // Make sure the route matches your backend route
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const imageDisplay = document.getElementById('imageDisplay');
        const predictionResult = document.getElementById('predictionResult');
        imageDisplay.src = URL.createObjectURL(file);
        imageDisplay.style.display = 'block';
        predictionResult.textContent = `Predicted class: ${data.prediction}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
