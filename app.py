from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

model = None
scaler = StandardScaler()
label_encoders = {}
categorical_columns = []
class_weights = {}

def load_model_and_preprocess():
    global model, scaler, label_encoders, categorical_columns, class_weights
    
    # Load the model
    model = keras.models.load_model('credit_card_approval_model.h5')
    
    # Load and preprocess your data
    data = pd.read_csv('Cleaned_Data.csv')  # Replace with your actual file
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    X = data.drop('Status', axis=1).drop('Applicant_ID', axis=1)
    y = data['Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

load_model_and_preprocess()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    form_data = request.form
    
    # Convert form data to DataFrame
    data = pd.DataFrame([form_data])
    for col in categorical_columns:
        if col in data.columns:
            data[col] = label_encoders[col].fit_transform(data[col].astype(str))

    
    # Preprocess the features
    # Load the scaler used during training
    X = scaler.transform(data)

    # Make prediction
    output = (model.predict(X) > 0.5).astype(int)

    # Return the result
    return render_template('predict.html', prediction_text='Credit Card Approval Prediction: {}'.format('Approved' if output[0][0] else 'Rejected'))

if __name__ == "__main__":
    app.run(debug=True)