import pandas as pd
from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

data = pd.read_csv('mushrooms2.csv', encoding='utf-8')
labelencoder=LabelEncoder()
for column in data.columns:
    data[column] = labelencoder.fit_transform(data[column])

X = data.drop(['class'], axis=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():


    feature_values = request.form.to_dict()
    model_,model_name = feature_values.popitem()
    if model_name == 's':
        model_filename= 'svc_mushroom.pkl'
    # elif model_name == 'a':
    #     model_filename = 'randomforest.pkl'
    # elif model_name == 'd':
    #     model_filename = 'decisiontree.pkl'
    elif model_name == 'l':
        model_filename = 'logistic_regression.pkl'
    elif model_name == 'k':
        model_filename = 'knn.pkl'

    loaded_model = joblib.load(model_filename)
    input_encoded = pd.get_dummies(feature_values)
    input_selected = input_encoded.reindex(columns=X.columns, fill_value=0)
    # Use the loaded model to make predictions
    classification = loaded_model.predict(input_selected)
    print(f"Classification: {classification}")
    prediction_label = "Edible" if classification[0] == 1 else "Not Edible"
    return render_template('index.html', prediction_label=prediction_label)

if __name__ == '__main__':
    # app.run(host='0.0.0.0',port=8080)
    app.run()
