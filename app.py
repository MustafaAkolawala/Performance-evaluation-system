from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import csv

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )

        if data.race_ethnicity == "group A":
                race_ethnicity = "Punjabi"
        if data.race_ethnicity == "group B":
                race_ethnicity = "Bengali"
        if data.race_ethnicity == "group C":
                race_ethnicity = "Marathi"
        if data.race_ethnicity == "group D":
                race_ethnicity = "Tamil"
        if data.race_ethnicity == "group E":
                race_ethnicity = "Kashmiri"

        with open('artifacts./savedinfo.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                data.gender,
                race_ethnicity,
                data.parental_level_of_education,
                data.lunch,
                data.test_preparation_course,
                data.reading_score,
                data.writing_score
            ])

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("after prediction")

        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)