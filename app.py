from flask import Flask, jsonify, render_template, request, send_file
from gevent.pywsgi import WSGIServer
import cv2
import io

from open_cv import open_cv
from SegmentationModelPyTorch.model_test import pytorch
from SegmentationPyTorchPredict.predict import pytorch_image_models
from PIL import Image

app = Flask(__name__)


def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/check_image', methods=['POST'])
def check_image():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file_to_check = request.files['file']

    if file_to_check.filename == '':
        return "No selected file", 400

    if not allowed_file(file_to_check.filename):
        return "Only image files are allowed", 400

    method = request.form['method']
    image = None
    if method:
        if method == 'OpenCV-Python':
            result = open_cv(file_to_check)
            _, buffer = cv2.imencode('.png', result)
            image = io.BytesIO(buffer)
        elif method == 'PyTorch':
            image = pytorch(file_to_check)
        elif method == 'PyTorch Image Models (timm)':
            result_image = pytorch_image_models(file_to_check)
            img_io = io.BytesIO()
            result_image.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')

        return send_file(image, mimetype='image/png')
    return "Method not specified", 400


if __name__ == "__main__":
    print('Flask is running')
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()