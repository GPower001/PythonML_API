# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix
# import numpy as np
# import os
# import pickle

# # --- Global variables for the model, encoder, and dataframes ---
# svc_model = None
# label_encoder = None
# all_symptoms_columns = []  # To store the order of symptom columns from Training.csv

# # DataFrames for recommendation system (loaded once)
# sym_def = None
# precautions_df = None
# workout_df = None
# description_df = None
# medications_df = None
# diets_df = None

# def load_data(filepath='Training.csv'):
#     """Load and preprocess the training data."""
#     global all_symptoms_columns
    
#     data = pd.read_csv(filepath)
#     # Ensure column names are properly formatted (lowercase with spaces)
#     data.columns = data.columns.str.strip().str.lower()
    
#     # Ensure all symptom columns are numeric (0 or 1)
#     symptom_cols = [col for col in data.columns if col != 'prognosis']
#     for col in symptom_cols:
#         data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    
#     X = data.drop('prognosis', axis=1)
#     y = data['prognosis'].astype(str).str.strip()  # Ensure prognosis is clean string
    
#     # Store symptom columns for later use
#     all_symptoms_columns = X.columns.tolist()
    
#     return X, y

# def normalize_symptom_input(symptom_name):
#     """Convert input symptom names to match training data format"""
#     return symptom_name.strip().lower().replace('_', ' ')

# def train_model(X_train, y_train):
#     """Train and return the SVC model."""
#     model = SVC(kernel='linear')
#     model.fit(X_train, y_train)
#     return model

# def evaluate_model(model, X_test, y_test):
#     """Evaluate model performance."""
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     cm = confusion_matrix(y_test, predictions)
#     return accuracy, cm

# def save_model(model, filename='svc.pkl'):
#     """Save the trained model to disk."""
#     with open(filename, 'wb') as f:
#         pickle.dump(model, f)

# def load_saved_model(filename='svc.pkl'):
#     """Load a saved model from disk."""
#     with open(filename, 'rb') as f:
#         return pickle.load(f)

# def load_recommendation_data():
#     """Load all supplementary dataframes with proper cleaning"""
#     global sym_def, precautions_df, workout_df, description_df, medications_df, diets_df
    
#     # Load each CSV with proper cleaning
#     def clean_df(df):
#         df.columns = df.columns.str.strip()
#         if 'Disease' in df.columns:
#             df['Disease'] = df['Disease'].astype(str).str.strip().str.lower()
#         return df
    
#     sym_def = clean_df(pd.read_csv("symtoms_df.csv"))
#     precautions_df = clean_df(pd.read_csv("precautions_df.csv"))
#     workout_df = clean_df(pd.read_csv("workout_df.csv"))
#     description_df = clean_df(pd.read_csv("description.csv"))
#     medications_df = clean_df(pd.read_csv("medications.csv"))
#     diets_df = clean_df(pd.read_csv("diets.csv"))
    
#     # Strip whitespace from relevant columns
#     for df in [sym_def, precautions_df, workout_df, description_df, medications_df, diets_df]:
#         df.columns = df.columns.str.strip()
#         if 'Disease' in df.columns:
#             df['Disease'] = df['Disease'].astype(str).str.strip()

# def load_and_train_model():
#     """
#     Main function to load all data, train the model, and prepare recommendation system.
#     This should be called once when the application starts.
#     """
#     global svc_model, label_encoder
    
#     print("--- Starting load_and_train_model ---")
    
#     try:
#         # Load and preprocess training data
#         X, y = load_data('Training.csv')
        
#         # Initialize and fit the LabelEncoder
#         label_encoder = LabelEncoder()
#         y_encoded = label_encoder.fit_transform(y)
        
#         # Split data for training and evaluation
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y_encoded, test_size=0.3, random_state=20
#         )
        
#         # Train the model
#         svc_model = train_model(X_train, y_train)
        
#         # Evaluate model
#         accuracy, cm = evaluate_model(svc_model, X_test, y_test)
#         print(f"Model trained with accuracy: {accuracy:.2f}")
#         print("Confusion Matrix:\n", cm)
        
#         # Load recommendation data
#         load_recommendation_data()
#         print("Recommendation data loaded successfully.")
        
#         # Save the trained model for future use
#         save_model(svc_model)
#         print("Model saved to disk.")
        
#         print("--- Finished load_and_train_model ---")
        
#     except Exception as e:
#         print(f"ERROR during model/data loading or training: {e}", exc_info=True)
#         raise

# def get_predicted_value(patient_symptoms: dict) -> str:
#     """
#     Predicts the disease based on the patient's symptoms.
    
#     Args:
#         patient_symptoms (dict): Dictionary where keys are symptom names and values are 1 (present) or 0 (absent).
        
#     Returns:
#         str: The predicted disease name.
#     """
#     print("--- Starting get_predicted_value ---")
#     if svc_model is None or label_encoder is None or not all_symptoms_columns:
#         print("Model or encoder not loaded. Attempting to load/train.")
#         load_and_train_model()
#         if svc_model is None:
#             raise RuntimeError("Model could not be loaded or trained.")

#     # Normalize input symptoms to match training data format
#     processed_symptoms = {}
#     for symptom, value in patient_symptoms.items():
#         norm_symptom = normalize_symptom_input(symptom)
#         if norm_symptom in all_symptoms_columns:
#             processed_symptoms[norm_symptom] = 1 if str(value).strip() == '1' else 0
#         else:
#             print(f"Note: Symptom '{symptom}' not found in trained symptoms. Ignoring.")
    
#     # Create input array with all symptoms initialized to 0
#     input_features = np.zeros(len(all_symptoms_columns))
    
#     # Map provided symptoms to the correct columns
#     for i, symptom_col in enumerate(all_symptoms_columns):
#         if symptom_col in processed_symptoms and processed_symptoms[symptom_col] == 1:
#             input_features[i] = 1

#     # Make prediction
#     prediction_input = input_features.astype(float).reshape(1, -1)
#     predicted_label_encoded = svc_model.predict(prediction_input)
#     predicted_disease = label_encoder.inverse_transform(predicted_label_encoded)[0]

#     print(f"Predicted disease: {predicted_disease}")
#     print("--- Finished get_predicted_value ---")
#     return predicted_disease

# def get_recommendations(disease: str) -> dict:
#     """
#     Provides complete recommendations for a given disease.
#     """
#     recommendations = {
#         "description": "",
#         "precautions": [],
#         "medications": [],
#         "workout": [],
#         "diet": []
#     }

#     cleaned_disease = disease.strip().lower()
    
#     # Get description
#     desc_row = description_df[
#         description_df['Disease'].str.strip().str.lower() == cleaned_disease
#     ]
#     if not desc_row.empty:
#         recommendations["description"] = desc_row['Description'].iloc[0]

#     # Get precautions (4 items)
#     prec_row = precautions_df[
#         precautions_df['Disease'].str.strip().str.lower() == cleaned_disease
#     ]
#     if not prec_row.empty:
#         recommendations["precautions"] = [
#             prec_row[f'Precaution_{i}'].iloc[0].strip() 
#             for i in range(1,5) 
#             if pd.notna(prec_row[f'Precaution_{i}'].iloc[0])
#         ]

#     # Get medications (4 items)
#     med_row = medications_df[
#         medications_df['Disease'].str.strip().str.lower() == cleaned_disease
#     ]
#     if not med_row.empty:
#         recommendations["medications"] = [
#             med_row[f'Medication_{i}'].iloc[0].strip()
#             for i in range(1,5)
#             if pd.notna(med_row[f'Medication_{i}'].iloc[0])
#         ]

#     # Get workout
#     workout_row = workout_df[
#         workout_df['disease'].str.strip().str.lower() == cleaned_disease
#     ]
#     if not workout_row.empty and pd.notna(workout_row['workout'].iloc[0]):
#         recommendations["workout"] = [
#             w.strip() for w in workout_row['workout'].iloc[0].split(',')
#         ]

#     # Get diet (4 items)
#     diet_row = diets_df[
#         diets_df['Disease'].str.strip().str.lower() == cleaned_disease
#     ]
#     if not diet_row.empty:
#         recommendations["diet"] = [
#             diet_row[f'Diet_{i}'].iloc[0].strip()
#             for i in range(1,5)
#             if pd.notna(diet_row[f'Diet_{i}'].iloc[0])
#         ]

#     return recommendations

# def get_available_symptoms():
#     """Return list of all symptoms the model recognizes"""
#     return sorted(all_symptoms_columns)

# # Example usage for testing
# if __name__ == "__main__":
#     load_and_train_model()
#     print("Available symptoms:", get_available_symptoms())

#     # Test prediction with different symptom formats
#     sample_symptoms = {
#         'itching': 1,
#         'skin rash': 1,  # with space
#         'nodal_skin_eruptions': 1,  # with underscore
#         'HIGH FEVER': 0,  # uppercase
#         'unknown_symptom': 1  # will be ignored
#     }

#     predicted_disease = get_predicted_value(sample_symptoms)
#     print(f"\nPredicted Disease: {predicted_disease}")
#     print("Recommendations:", get_recommendations(predicted_disease))




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pickle

# --- Global variables for the model, encoder, and dataframes ---
svc_model = None
label_encoder = None
all_symptoms_columns = []

# DataFrames for recommendation system
sym_def = None
precautions_df = None
workout_df = None
description_df = None
diets_df = None

def load_data(filepath='Training.csv'):
    global all_symptoms_columns
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip().str.lower()
    
    symptom_cols = [col for col in data.columns if col != 'prognosis']
    for col in symptom_cols:
        data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)

    X = data.drop('prognosis', axis=1)
    y = data['prognosis'].astype(str).str.strip()
    all_symptoms_columns = X.columns.tolist()
    return X, y

def normalize_symptom_input(symptom_name):
    return symptom_name.strip().lower().replace('_', ' ')

def train_model(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, cm

def save_model(model, filename='svc.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_saved_model(filename='svc.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_recommendation_data():
    global sym_def, precautions_df, workout_df, description_df, diets_df

    def clean_df(df):
        df.columns = df.columns.str.strip()
        if 'Disease' in df.columns:
            df['Disease'] = df['Disease'].astype(str).str.strip().str.lower()
        return df

    sym_def = clean_df(pd.read_csv("symtoms_df.csv"))
    precautions_df = clean_df(pd.read_csv("precautions_df.csv"))
    workout_df = clean_df(pd.read_csv("workout_df.csv"))
    description_df = clean_df(pd.read_csv("description.csv"))
    diets_df = clean_df(pd.read_csv("diets.csv"))

    for df in [sym_def, precautions_df, workout_df, description_df, diets_df]:
        df.columns = df.columns.str.strip()
        if 'Disease' in df.columns:
            df['Disease'] = df['Disease'].astype(str).str.strip()

def load_and_train_model():
    global svc_model, label_encoder
    print("--- Starting load_and_train_model ---")

    try:
        X, y = load_data('Training.csv')
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=20
        )

        svc_model = train_model(X_train, y_train)
        accuracy, cm = evaluate_model(svc_model, X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print("Confusion Matrix:\n", cm)

        load_recommendation_data()
        print("Recommendation data loaded successfully.")
        save_model(svc_model)

    except Exception as e:
        print(f"ERROR during model/data loading or training: {e}")
        raise

def get_predicted_value(patient_symptoms: dict) -> str:
    print("--- Starting get_predicted_value ---")

    if svc_model is None or label_encoder is None or not all_symptoms_columns:
        print("Model or encoder not loaded. Attempting to load/train.")
        load_and_train_model()

    processed_symptoms = {}
    for symptom, value in patient_symptoms.items():
        norm_symptom = normalize_symptom_input(symptom)
        if norm_symptom in all_symptoms_columns:
            processed_symptoms[norm_symptom] = 1 if str(value).strip() == '1' else 0
        else:
            print(f"Note: Symptom '{symptom}' not found in trained symptoms. Ignoring.")

    input_features = np.zeros(len(all_symptoms_columns))
    for i, symptom_col in enumerate(all_symptoms_columns):
        if symptom_col in processed_symptoms and processed_symptoms[symptom_col] == 1:
            input_features[i] = 1

    prediction_input = input_features.astype(float).reshape(1, -1)
    predicted_label_encoded = svc_model.predict(prediction_input)
    predicted_disease = label_encoder.inverse_transform(predicted_label_encoded)[0]

    print(f"Predicted disease: {predicted_disease}")
    print("--- Finished get_predicted_value ---")
    return predicted_disease

def get_recommendations(disease: str) -> dict:
    recommendations = {
        "description": "",
        "precautions": [],
        "workout": [],
        "diet": []
    }

    cleaned_disease = disease.strip().lower()

    # Description
    desc_row = description_df[description_df['Disease'].str.strip().str.lower() == cleaned_disease]
    if not desc_row.empty:
        recommendations["description"] = desc_row['Description'].iloc[0]

    # Precautions
    prec_row = precautions_df[precautions_df['Disease'].str.strip().str.lower() == cleaned_disease]
    if not prec_row.empty:
        for i in range(1, 5):
            col = f'Precaution_{i}'
            if col in prec_row.columns and pd.notna(prec_row[col].iloc[0]):
                recommendations["precautions"].append(prec_row[col].iloc[0].strip())

    # Workout
    workout_row = workout_df[workout_df['disease'].str.strip().str.lower() == cleaned_disease]
    if not workout_row.empty and pd.notna(workout_row['workout'].iloc[0]):
        recommendations["workout"] = [w.strip() for w in workout_row['workout'].iloc[0].split(',')]

    # Diet
    diet_row = diets_df[diets_df['Disease'].str.strip().str.lower() == cleaned_disease]
    if not diet_row.empty:
        for i in range(1, 5):
            col = f'Diet_{i}'
            if col in diet_row.columns and pd.notna(diet_row[col].iloc[0]):
                recommendations["diet"].append(diet_row[col].iloc[0].strip())

    return recommendations

def get_available_symptoms():
    return sorted(all_symptoms_columns)

# Example usage for testing
if __name__ == "__main__":
    load_and_train_model()
    print("Available symptoms:", get_available_symptoms())

    sample_symptoms = {
        'itching': 1,
        'skin rash': 1,
        'nodal_skin_eruptions': 1,
        'HIGH FEVER': 0,
        'unknown_symptom': 1
    }

    predicted_disease = get_predicted_value(sample_symptoms)
    print(f"\nPredicted Disease: {predicted_disease}")
    print("Recommendations:", get_recommendations(predicted_disease))
