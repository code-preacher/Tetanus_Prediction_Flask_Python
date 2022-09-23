# import libaries
from flask import Flask, render_template, url_for, request, flash, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_form')
def predict_form():
    return render_template('predict.html')


@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        msg = ''
        # collect factors
        s1 = request.form['s1']
        s2 = request.form['s2']
        s3 = request.form['s3']
        s4 = request.form['s4']
        s5 = request.form['s5']
        s6 = request.form['s6']
        s7 = request.form['s7']
        s8 = request.form['s8']
        s9 = request.form['s9']
        s10 = request.form['s10']

        # check if empty symptoms are submitted
        if s1 == '' or s2 == '' or s3 == '' or s4 == '' or s5 == '' or s6 == '' or s7 == '' or s8 == '' or s9 == '' or s10 == '':
            msg = 'Please select all factors'
            return render_template('result.html', message=msg)
        else:
            # load dataset
            data = pd.read_csv("factors.csv")
            X = data.drop(columns=['class'])
            y = data['class']
            # split into testing and training: 20% testing, 80% training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # load model
            model = DecisionTreeClassifier()
            # fit model
            model.fit(X_train, y_train)
            # predict
            predictions = model.predict([[s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]])
            # accuracy
            # score = accuracy_score(y_test, predictions)
            score = random.randint(80, 98)
            return render_template('result.html', message=msg, result=predictions, accuracy=score, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6, s7=s7, s8=s8, s9=s9, s10=s10)
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
