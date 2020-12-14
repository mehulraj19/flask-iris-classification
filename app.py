from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def resultForm():
    sepalLength = request.form['sepalLength']
    sepalWidth = request.form['sepalWidth']
    petalLength = request.form['petalLength']
    petalWidth = request.form['petalWidth']
    
    arr = np.array([[sepalLength, sepalWidth, petalLength, petalWidth]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)

if __name__ == "__main__":
    app.run(port=3000,debug=True)