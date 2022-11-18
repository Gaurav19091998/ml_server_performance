from flask import Flask, render_template, url_for, request
import numpy as np

import pickle

# from pyexpat import model

app = Flask(__name__)

model = pickle.load(open('predicts.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_value():
    default_value = '0'
    memory = request.form.get('memory', default_value)
    disk = request.form.get('disk', default_value)
    cpu = request.form.get('cpu', default_value)
    time = float(request.form.get('time', default_value))
    inputs = np.array([[memory, disk, cpu]])
    result = model.predict(inputs)
    unit_per = result/5
    data = unit_per*time
    # data = (result / 5) * time

    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
