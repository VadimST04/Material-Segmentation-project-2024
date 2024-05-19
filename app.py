from flask import Flask, jsonify, render_template, request, send_file
from gevent.pywsgi import WSGIServer
import cv2
import io

from open_cv import open_cv
from pytorch import pytorch
from pytorch_image_models import pytorch_image_models

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/check_image', methods=['POST'])
def check_image():
    file_to_check = request.files['file']
    method = request.form['method']
    result = 'result'
    if method:
        if method == 'OpenCV-Python':
            result = open_cv(file_to_check)
        elif method == 'PyTorch':
            result = pytorch(file_to_check)
        elif method == 'PyTorch Image Models (timm)':
            result = pytorch_image_models(file_to_check)

        _, buffer = cv2.imencode('.png', result)
        io_buf = io.BytesIO(buffer)
        return send_file(io_buf, mimetype='image/png')
    return "Method not specified", 400


if __name__ == "__main__":
    print('Flask is running')
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()