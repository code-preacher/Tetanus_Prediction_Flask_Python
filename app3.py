from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_form')
def predict_form():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv("factors.csv")

    X = data.drop(['class'], axis=1)
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

    estimator = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    selectFeatures = RFE(estimator, n_features_to_select=10)
    selectFeatures.fit(X_train, y_train)

    Xtrain = selectFeatures.transform(X_train)
    Xtest = selectFeatures.transform(X_test)

    model = LogisticRegression()
    model.fit(Xtrain, y_train)

    if request.method == 'POST':
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

        test = np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10], dtype='float64')
        test = test.reshape(1, -1)
        test = test.astype(float)
        my_prediction = model.predict(test)
    return render_template("result.html", result=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
