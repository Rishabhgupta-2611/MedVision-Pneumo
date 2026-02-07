from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from utils.model_loader import load_selected_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    file = request.files.get('image')
    model_choice = request.form.get('model', 'model1')

    if file is None or file.filename == '':
        error_msg = 'No file provided. Please upload a chest X-ray image.'
        return render_template('index.html', error=error_msg)

    if not allowed_file(file.filename):
        error_msg = 'Unsupported file type. Please upload JPG or PNG images only.'
        return render_template('index.html', error=error_msg)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    try:
        model = load_selected_model(model_choice)
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        result = 'Pneumonia Detected' if confidence > 0.5 else 'Normal'

        return render_template(
            'result.html',
            result=result,
            confidence=round(confidence * 100, 2),
            image_path=filepath,
            model=model_choice
        )
    except Exception as e:
        error_msg = f'Prediction failed: {str(e)}'
        return render_template('error.html', error=error_msg), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    file = request.files.get('image')
    model_choice = request.form.get('model', 'model1')

    if file is None or file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file provided.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Unsupported file type.'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    try:
        model = load_selected_model(model_choice)
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        result = 'Pneumonia Detected' if confidence > 0.5 else 'Normal'

        return jsonify({
            'status': 'success',
            'result': result,
            'confidence': round(confidence * 100, 2),
            'image_path': filepath,
            'model': model_choice
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Prediction failed: {str(e)}'}), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', error=str(e)), 500


if __name__ == '__main__':
    app.run(debug=True)
