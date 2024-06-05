document.addEventListener('DOMContentLoaded', function () {
    var fileInput = document.getElementById('fileInput');
    var displayImage = document.querySelector('img[alt="First image"]');

    fileInput.addEventListener('change', function(event) {
        var files = event.target.files;
        if (files && files[0]) {
            var reader = new FileReader();

            reader.onload = function(e) {
                displayImage.src = e.target.result;
            };

            reader.readAsDataURL(files[0]);
        }
    });
});

document.getElementById('methodSelect').addEventListener('change', function() {
    var methodName = document.querySelector('.method-name');
    methodName.textContent = this.value;
});

document.getElementById('imageForm').addEventListener('submit', function(event) {
    event.preventDefault();

    file = document.getElementById('fileInput').files[0]
    if (!file) {
        alert("Select a file first.");
    } else {
        processRawFile(file)
    }
});

function processRawFile(file) {
    var spinner = document.getElementById('spinner');
    var imageDisplay = document.getElementById('imageDisplay');

    const formData = new FormData();

    formData.append('file', file);
    formData.append('method', document.getElementById('methodSelect').value);

    imageDisplay.style.display = 'none';
    spinner.style.display = 'block';

    fetch('/check_image', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.status === 400) {
            return response.text().then(text => { throw new Error(text); });
        }
        return response.blob();
    })
    .then(blob => {
        var url = URL.createObjectURL(blob);
        imageDisplay.src = url;

        spinner.style.display = 'none';
        imageDisplay.style.display = 'block';
    })
    .catch((error) => {
        spinner.style.display = 'none';
        alert('Error: ' + error.message);
    });
}