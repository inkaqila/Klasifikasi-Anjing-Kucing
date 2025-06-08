from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import base64
import io
import uuid

app = Flask(__name__)
model = joblib.load("model.pkl")
class_names = ['Kucing', 'Anjing']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img).flatten()
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    prediction_proba = model.predict_proba([img_array])[0]
    predicted_class = class_names[np.argmax(prediction_proba)]

    # Dummy confusion matrix for display (optional: replace with real tracking if needed)
    dummy_cm = np.array([[45, 5], [3, 47]])
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=dummy_cm, display_labels=class_names)
    disp.plot(ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    prob_dict = {class_names[i]: float(prediction_proba[i]) for i in range(len(class_names))}

    return render_template(
        'result.html',
        prediction=predicted_class,
        probabilities=prob_dict,
        confusion_image=image_base64,
        uploaded_image=filename
    )

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

