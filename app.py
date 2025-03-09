from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

class InputData:
    def __init__(self, Country, Age, Gender, Education_Level, BMI, Physical_Activity_Level, 
                 Smoking_Status, Alcohol_Consumption, Diabetes, Hypertension, Cholesterol_Level,
                 Family_History_of_Alzheimers, Cognitive_Test_Score, Depression_Level, 
                 Sleep_Quality, Dietary_Habits, Air_Pollution_Exposure, Employment_Status,
                 Marital_Status, Genetic_Risk_Factor, Social_Engagement_Level, Income_Level,
                 Stress_Levels, Urban_vs_Rural_Living):
        self.Country = Country
        self.Age = Age
        self.Gender = Gender
        self.Education_Level = Education_Level
        self.BMI = BMI
        self.Physical_Activity_Level = Physical_Activity_Level
        self.Smoking_Status = Smoking_Status
        self.Alcohol_Consumption = Alcohol_Consumption
        self.Diabetes = Diabetes
        self.Hypertension = Hypertension
        self.Cholesterol_Level = Cholesterol_Level
        self.Family_History_of_Alzheimers = Family_History_of_Alzheimers
        self.Cognitive_Test_Score = Cognitive_Test_Score
        self.Depression_Level = Depression_Level
        self.Sleep_Quality = Sleep_Quality
        self.Dietary_Habits = Dietary_Habits
        self.Air_Pollution_Exposure = Air_Pollution_Exposure
        self.Employment_Status = Employment_Status
        self.Marital_Status = Marital_Status
        self.Genetic_Risk_Factor = Genetic_Risk_Factor
        self.Social_Engagement_Level = Social_Engagement_Level
        self.Income_Level = Income_Level
        self.Stress_Levels = Stress_Levels
        self.Urban_vs_Rural_Living = Urban_vs_Rural_Living
    
    def to_dict(self):
        return {
            'Country': self.Country,
            'Age': self.Age,
            'Gender': self.Gender,
            'Education Level': self.Education_Level,
            'BMI': self.BMI,
            'Physical Activity Level': self.Physical_Activity_Level,
            'Smoking Status': self.Smoking_Status,
            'Alcohol Consumption': self.Alcohol_Consumption,
            'Diabetes': self.Diabetes,
            'Hypertension': self.Hypertension,
            'Cholesterol Level': self.Cholesterol_Level,
            'Family History of Alzheimer’s': self.Family_History_of_Alzheimers,
            'Cognitive Test Score': self.Cognitive_Test_Score,
            'Depression Level': self.Depression_Level,
            'Sleep Quality': self.Sleep_Quality,
            'Dietary Habits': self.Dietary_Habits,
            'Air Pollution Exposure': self.Air_Pollution_Exposure,
            'Employment Status': self.Employment_Status,
            'Marital Status': self.Marital_Status,
            'Genetic Risk Factor (APOE-ε4 allele)': self.Genetic_Risk_Factor,
            'Social Engagement Level': self.Social_Engagement_Level,
            'Income Level': self.Income_Level,
            'Stress Levels': self.Stress_Levels,
            'Urban vs Rural Living': self.Urban_vs_Rural_Living
        }

model = joblib.load('models/model.pkl')

@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"health_check": "OK", "model_version": 1})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Create InputData instance from request JSON
    input_data = InputData(
        Country=data.get('Country'),
        Age=data.get('Age'),
        Gender=data.get('Gender'),
        Education_Level=data.get('Education Level'),
        BMI=data.get('BMI'),
        Physical_Activity_Level=data.get('Physical Activity Level'),
        Smoking_Status=data.get('Smoking Status'),
        Alcohol_Consumption=data.get('Alcohol Consumption'),
        Diabetes=data.get('Diabetes'),
        Hypertension=data.get('Hypertension'),
        Cholesterol_Level=data.get('Cholesterol Level'),
        Family_History_of_Alzheimers=data.get('Family History of Alzheimer’s'),
        Cognitive_Test_Score=data.get('Cognitive Test Score'),
        Depression_Level=data.get('Depression Level'),
        Sleep_Quality=data.get('Sleep Quality'),
        Dietary_Habits=data.get('Dietary Habits'),
        Air_Pollution_Exposure=data.get('Air Pollution Exposure'),
        Employment_Status=data.get('Employment Status'),
        Marital_Status=data.get('Marital Status'),
        Genetic_Risk_Factor=data.get('Genetic Risk Factor (APOE-ε4 allele)'),
        Social_Engagement_Level=data.get('Social Engagement Level'),
        Income_Level=data.get('Income Level'),
        Stress_Levels=data.get('Stress Levels'),
        Urban_vs_Rural_Living=data.get('Urban vs Rural Living')
    )
    
    # Convert to DataFrame for prediction
    df = pd.DataFrame([input_data.to_dict().values()], 
                      columns=input_data.to_dict().keys())
    pred = model.predict(df)
    return jsonify({"predicted_class": int(pred[0])})

if __name__ == "__main__":
    app.run(debug=True)


