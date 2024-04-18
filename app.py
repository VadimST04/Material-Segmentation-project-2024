from flask import Flask, jsonify, render_template, request
from gevent.pywsgi import WSGIServer

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
    return jsonify({
        'result': result
    })


if __name__ == "__main__":
    print('Flask is running')
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()