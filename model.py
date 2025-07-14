# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# import numpy as np
# import os
# import pickle

# # --- Global variables for the model, encoder, and dataframes ---
# svc_model = None
# label_encoder = None
# all_symptoms_columns = [] # To store the order of symptom columns from Training.csv

# # DataFrames for recommendation system (loaded once)
# sym_def = None
# precautions_df = None
# workout_df = None
# description_df = None
# medications_df = None
# diets_df = None

# def load_and_train_model():
#     """
#     Loads all necessary data, trains the SVC model, fits the LabelEncoder,
#     and loads supplementary dataframes for recommendations.
#     This function will be called once when the Flask app starts.
#     """
#     global svc_model, label_encoder, all_symptoms_columns
#     global sym_def, precautions_df, workout_df, description_df, medications_df, diets_df

#     try:
#         print("--- Starting load_and_train_model ---")
#         # --- Load Training Data for ML Model ---
#         datasets = pd.read_csv('Training.csv')
#         print(f"Original Training.csv shape: {datasets.shape}")

#         # Aggressively strip whitespace from all column names
#         datasets.columns = datasets.columns.str.strip()
#         print(f"Cleaned Training.csv columns: {datasets.columns.tolist()}")

#         # Ensure all symptom columns are numeric (0 or 1)
#         symptom_cols = datasets.columns.drop('prognosis', errors='ignore').tolist()
#         print(f"Symptom columns identified: {len(symptom_cols)}")

#         # --- NEW AGGRESSIVE CLEANING FOR SYMPTOM COLUMNS ---
#         for col in symptom_cols:
#             # Apply a lambda function to each cell to force it to 0 or 1
#             # Using errors='coerce' in to_numeric to turn unparseable values into NaN
#             # Then fill NaN with 0 and convert to int
#             datasets[col] = pd.to_numeric(datasets[col].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
#             print(f"  Aggressively cleaned column '{col}'. Unique values: {datasets[col].unique().tolist()}")

#         # Separate features (X) and target (y)
#         X = datasets.drop('prognosis', axis=1)
#         y = datasets['prognosis']
#         print(f"X shape after drop: {X.shape}, y shape: {y.shape}")

#         # --- CRITICAL FIX: Ensure prognosis column is explicitly string type before LabelEncoder ---
#         # Convert the 'prognosis' series to string type, then strip whitespace from each element.
#         # This is the most robust way to ensure LabelEncoder only sees strings.
#         y = datasets['prognosis'].astype(str).str.strip()
#         print(f"Unique prognosis values after stripping and explicit string conversion: {y.unique().tolist()}")
#         print(f"y Series dtype before LabelEncoder fit: {y.dtype}")


#         # Explicitly convert X to float type for consistency with sklearn expectations
#         # Adding a more robust check for non-numeric values before final conversion
#         for col in X.columns:
#             if not pd.api.types.is_numeric_dtype(X[col]):
#                 print(f"Warning: Column '{col}' in X is not numeric before final float conversion. Sample values: {X[col].head().tolist()}")
#                 # Attempt to convert again, coercing errors
#                 X[col] = pd.to_numeric(X[col], errors='coerce')
#         X = X.astype(float)
#         print(f"X dtypes after final float conversion (sample): {X.dtypes.head()}")
#         if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
#             print("CRITICAL ERROR: X still contains non-numeric columns after all cleaning attempts!")
#             # Raise an error to stop execution if data is still problematic
#             raise ValueError("X DataFrame contains non-numeric data after aggressive cleaning.")


#         # Store all symptom columns for consistent input formatting during prediction
#         all_symptoms_columns = X.columns.tolist()
#         print(f"Total symptoms columns stored: {len(all_symptoms_columns)}")

#         # Initialize and fit the LabelEncoder for disease names
#         # LabelEncoder is still useful for consistency, even if not used for inverse_transform here
#         label_encoder = LabelEncoder()
#         label_encoder.fit(y)
#         print(f"LabelEncoder fitted with {len(label_encoder.classes_)} classes.")
#         print(f"LabelEncoder classes_ dtype: {label_encoder.classes_.dtype}")
#         print(f"LabelEncoder classes_ sample: {label_encoder.classes_[:5].tolist()}")
#         # --- NEW: Verify all classes are strings ---
#         if not all(isinstance(cls, str) for cls in label_encoder.classes_):
#             print("CRITICAL WARNING: LabelEncoder classes_ contains non-string elements!")
#             problematic_classes = [cls for cls in label_encoder.classes_ if not isinstance(cls, str)]
#             print(f"Problematic classes: {problematic_classes}")


#         # Train the SVC model
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#         print(f"Training SVC model with X_train shape: {X_train.shape}")
#         print(f"X_train dtypes before SVC fit (sample): {X_train.dtypes.head()}")
#         print(f"y_train dtypes before SVC fit: {y_train.dtype}") # This will be 'object' (string)

#         svc_model = SVC(kernel='linear')
#         svc_model.fit(X_train, y_train) # SVC is fitted with string labels

#         print("Disease prediction model trained successfully.")

#         # --- Load Supplementary DataFrames for Recommendation System ---
#         sym_def = pd.read_csv("symtoms_df.csv")
#         precautions_df = pd.read_csv("precautions_df.csv")
#         workout_df = pd.read_csv("workout_df.csv")
#         description_df = pd.read_csv("description.csv")
#         medications_df = pd.read_csv("medications.csv")
#         diets_df = pd.read_csv("diets.csv")

#         # Strip whitespace from relevant columns in recommendation dataframes
#         for df in [sym_def, precautions_df, workout_df, description_df, medications_df, diets_df]:
#             df.columns = df.columns.str.strip()
#             if 'Disease' in df.columns:
#                 df['Disease'] = df['Disease'].astype(str).str.strip() # Ensure Disease column is string


#         print("Supplementary dataframes for recommendations loaded successfully.")
#         print("--- Finished load_and_train_model ---")

#     except FileNotFoundError as e:
#         print(f"ERROR: A required CSV file was not found: {e}. Please ensure all CSVs are in the correct directory.")
#         exit()
#     except ValueError as e:
#         print(f"DATA ERROR: {e}. Please check your Training.csv for non-numeric values that cannot be coerced.")
#         exit()
#     except Exception as e:
#         print(f"CRITICAL ERROR during model/data loading or training: {e}", exc_info=True)
#         exit()

# def get_predicted_value(patient_symptoms: dict) -> str:
#     """
#     Predicts the disease based on the patient's symptoms.

#     Args:
#         patient_symptoms (dict): A dictionary where keys are symptom names
#                                  (e.g., 'itching', 'fever') and values are 1 (present) or 0 (absent).
#                                  This dictionary should ideally contain all 132 symptoms,
#                                  or the function will fill missing ones with 0.

#     Returns:
#         str: The predicted disease name.
#     """
#     print("--- Starting get_predicted_value ---")
#     if svc_model is None or label_encoder is None or not all_symptoms_columns:
#         print("Model or encoder not loaded. Attempting to load/train.")
#         load_and_train_model()
#         if svc_model is None:
#             print("ERROR: Model could not be loaded or trained after retry.")
#             raise RuntimeError("Model could not be loaded or trained.")

#     # Sanitize incoming patient_symptoms values to ensure they are integers (0 or 1)
#     sanitized_symptoms = {}
#     print(f"Received patient_symptoms (first 5 items): {list(patient_symptoms.items())[:5]}")

#     for symptom, value in patient_symptoms.items():
#         try:
#             # Convert to string, strip whitespace, then check if it's '1'
#             sanitized_symptoms[symptom.strip()] = 1 if str(value).strip() == '1' else 0
#         except Exception:
#             # Default to 0 if any unexpected error during sanitization
#             sanitized_symptoms[symptom.strip()] = 0

#     print(f"Sanitized symptoms (first 5 items): {list(sanitized_symptoms.items())[:5]}")
#     print(f"Number of sanitized symptoms: {len(sanitized_symptoms)}")


#     # Create an input array with all 132 symptoms, initialized to 0
#     input_features = np.zeros(len(all_symptoms_columns))
#     print(f"Expected number of symptom columns: {len(all_symptoms_columns)}")


#     # Populate the input_features array based on the provided sanitized_symptoms
#     for i, symptom_col in enumerate(all_symptoms_columns):
#         cleaned_symptom_col = symptom_col.strip() # Ensure consistency with column names
#         if cleaned_symptom_col in sanitized_symptoms and sanitized_symptoms[cleaned_symptom_col] == 1:
#             input_features[i] = 1

#     print(f"Input features array (first 10 values): {input_features[:10]}")
#     print(f"Input features array dtype: {input_features.dtype}")


#     # Explicitly convert the input features array to float before prediction
#     prediction_input = input_features.astype(float) # Ensure all elements are float
#     print(f"Prediction input array dtype after float conversion: {prediction_input.dtype}")
#     prediction_input = prediction_input.reshape(1, -1) # Reshape for prediction (1 sample, 132 features)
#     print(f"Prediction input shape: {prediction_input.shape}")

#     # Make prediction
#     try:
#         # --- CRITICAL FIX: SVC.predict() returns the string label directly when trained with strings ---
#         predicted_disease_array = svc_model.predict(prediction_input)
#         predicted_disease = predicted_disease_array[0] # Get the single predicted string label

#         print(f"Predicted disease (from SVC.predict): {predicted_disease}")
#         print(f"Predicted disease type: {type(predicted_disease)}")

#     except Exception as e:
#         print(f"ERROR during SVC prediction: {e}")
#         print(f"Type of prediction_input: {type(prediction_input)}")
#         print(f"Shape of prediction_input: {prediction_input.shape}")
#         print(f"Dtype of prediction_input: {prediction_input.dtype}")
#         print(f"Sample of prediction_input values: {prediction_input.flatten()[:10]}")
#         raise # Re-raise the exception to propagate it

#     # No need for label_encoder.inverse_transform as SVC.predict already returns the string
#     # predicted_disease = label_encoder.inverse_transform(predicted_label_encoded)

#     print(f"Final predicted disease: {predicted_disease}")
#     print("--- Finished get_predicted_value ---")
#     return predicted_disease

# def get_recommendations(disease: str) -> dict:
#     """
#     Provides recommendations (description, precautions, medications, diet, workout)
#     for a given disease.

#     Args:
#         disease (str): The predicted disease name.

#     Returns:
#         dict: A dictionary containing various recommendations.
#     """
#     print("--- Starting get_recommendations ---")
#     if description_df is None: # Ensure dataframes are loaded
#         print("Recommendation dataframes not loaded. Attempting to load/train.")
#         load_and_train_model() # This will load all dataframes if not already
#         if description_df is None:
#             print("ERROR: Recommendation dataframes could not be loaded after retry.")
#             raise RuntimeError("Recommendation dataframes could not be loaded.")

#     recommendations = {
#         "description": "",
#         "precautions": [],
#         "medications": [],
#         "workout": [],
#         "diet": []
#     }

#     # Clean disease names for consistent lookup (e.g., strip whitespace, lower case)
#     cleaned_disease = disease.strip().replace('_', ' ').lower()
#     print(f"Looking up recommendations for cleaned disease: '{cleaned_disease}'")

#     # Get description
#     desc_row = description_df[description_df['Disease'].str.strip().str.replace('_', ' ').str.lower() == cleaned_disease]
#     if not desc_row.empty:
#         recommendations["description"] = desc_row['Description'].iloc[0]
#         print(f"Found description: {recommendations['description'][:50]}...")

#     # Get precautions
#     prec_row = precautions_df[precautions_df['Disease'].str.strip().str.replace('_', ' ').str.lower() == cleaned_disease]
#     if not prec_row.empty:
#         for i in range(1, 5):
#             col_name = f'Precaution_{i}'
#             if col_name in prec_row.columns and pd.notna(prec_row[col_name].iloc[0]):
#                 recommendations["precautions"].append(prec_row[col_name].iloc[0])
#         print(f"Found precautions: {recommendations['precautions']}")

#     # Get medications
#     med_row = medications_df[medications_df['Disease'].str.strip().str.replace('_', ' ').str.lower() == cleaned_disease]
#     if not med_row.empty:
#         for i in range(1, 5):
#             col_name = f'Medication_{i}'
#             if col_name in med_row.columns and pd.notna(med_row[col_name].iloc[0]):
#                 recommendations["medications"].append(med_row[col_name].iloc[0])
#         print(f"Found medications: {recommendations['medications']}")

#     # Get workout
#     workout_row = workout_df[workout_df['Disease'].str.strip().str.replace('_', ' ').str.lower() == cleaned_disease]
#     if not workout_row.empty:
#         for i in range(1, 5):
#             col_name = f'Workout_{i}'
#             if col_name in workout_row.columns and pd.notna(workout_row[col_name].iloc[0]):
#                 recommendations["workout"].append(workout_row[col_name].iloc[0])
#         print(f"Found workout: {recommendations['workout']}")

#     # Get diets
#     diet_row = diets_df[diets_df['Disease'].str.strip().str.replace('_', ' ').str.lower() == cleaned_disease]
#     if not diet_row.empty:
#         for i in range(1, 5):
#             col_name = f'Diet_{i}'
#             if col_name in diet_row.columns and pd.notna(diet_row[col_name].iloc[0]):
#                 recommendations["diet"].append(diet_row[col_name].iloc[0])
#         print(f"Found diets: {recommendations['diet']}")

#     print("--- Finished get_recommendations ---")
#     return recommendations


# # Example usage (for testing the model script directly)
# if __name__ == "__main__":
#     load_and_train_model()

#     # Example patient symptoms (ensure these match your training data columns)
#     # You should provide all 132 symptoms, even if most are 0.
#     # For a real application, you'd have a UI to select symptoms.
#     sample_symptoms = {symptom: 0 for symptom in all_symptoms_columns}
#     sample_symptoms['itching'] = 1
#     sample_symptoms['skin rash'] = 1
#     sample_symptoms['nodal skin eruptions'] = 1

#     predicted_disease = get_predicted_value(sample_symptoms)
#     print(f"Predicted Disease for sample symptoms: {predicted_disease}")
#     recommendations = get_recommendations(predicted_disease)
#     print("Recommendations:", recommendations)

#     sample_symptoms_2 = {symptom: 0 for symptom in all_symptoms_columns}
#     sample_symptoms_2['high fever'] = 1
#     sample_symptoms_2['chills'] = 1
#     sample_symptoms_2['headache'] = 1
#     sample_symptoms_2['nausea'] = 1

#     predicted_disease_2 = get_predicted_value(sample_symptoms_2)
#     print(f"\nPredicted Disease for second sample: {predicted_disease_2}")
#     recommendations_2 = get_recommendations(predicted_disease_2)
#     print("Recommendations:", recommendations_2)

# models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os
import pickle

# --- Global variables for the model, encoder, and dataframes ---
svc_model = None
label_encoder = None
all_symptoms_columns = []  # To store the order of symptom columns from Training.csv

# DataFrames for recommendation system (loaded once)
sym_def = None
precautions_df = None
workout_df = None
description_df = None
medications_df = None
diets_df = None

def load_data(filepath='Training.csv'):
    """Load and preprocess the training data."""
    global all_symptoms_columns
    
    data = pd.read_csv(filepath)
    # Ensure column names are properly formatted (lowercase with spaces)
    data.columns = data.columns.str.strip().str.lower()
    
    # Ensure all symptom columns are numeric (0 or 1)
    symptom_cols = [col for col in data.columns if col != 'prognosis']
    for col in symptom_cols:
        data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    
    X = data.drop('prognosis', axis=1)
    y = data['prognosis'].astype(str).str.strip()  # Ensure prognosis is clean string
    
    # Store symptom columns for later use
    all_symptoms_columns = X.columns.tolist()
    
    return X, y

def normalize_symptom_input(symptom_name):
    """Convert input symptom names to match training data format"""
    return symptom_name.strip().lower().replace('_', ' ')

def train_model(X_train, y_train):
    """Train and return the SVC model."""
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, cm

def save_model(model, filename='svc.pkl'):
    """Save the trained model to disk."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_saved_model(filename='svc.pkl'):
    """Load a saved model from disk."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_recommendation_data():
    """Load all supplementary dataframes for recommendations."""
    global sym_def, precautions_df, workout_df, description_df, medications_df, diets_df
    
    sym_def = pd.read_csv("symtoms_df.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    workout_df = pd.read_csv("workout_df.csv")
    description_df = pd.read_csv("description.csv")
    medications_df = pd.read_csv("medications.csv")
    diets_df = pd.read_csv("diets.csv")
    
    # Strip whitespace from relevant columns
    for df in [sym_def, precautions_df, workout_df, description_df, medications_df, diets_df]:
        df.columns = df.columns.str.strip()
        if 'Disease' in df.columns:
            df['Disease'] = df['Disease'].astype(str).str.strip()

def load_and_train_model():
    """
    Main function to load all data, train the model, and prepare recommendation system.
    This should be called once when the application starts.
    """
    global svc_model, label_encoder
    
    print("--- Starting load_and_train_model ---")
    
    try:
        # Load and preprocess training data
        X, y = load_data('Training.csv')
        
        # Initialize and fit the LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data for training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=20
        )
        
        # Train the model
        svc_model = train_model(X_train, y_train)
        
        # Evaluate model
        accuracy, cm = evaluate_model(svc_model, X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print("Confusion Matrix:\n", cm)
        
        # Load recommendation data
        load_recommendation_data()
        print("Recommendation data loaded successfully.")
        
        # Save the trained model for future use
        save_model(svc_model)
        print("Model saved to disk.")
        
        print("--- Finished load_and_train_model ---")
        
    except Exception as e:
        print(f"ERROR during model/data loading or training: {e}", exc_info=True)
        raise

def get_predicted_value(patient_symptoms: dict) -> str:
    """
    Predicts the disease based on the patient's symptoms.
    
    Args:
        patient_symptoms (dict): Dictionary where keys are symptom names and values are 1 (present) or 0 (absent).
        
    Returns:
        str: The predicted disease name.
    """
    print("--- Starting get_predicted_value ---")
    if svc_model is None or label_encoder is None or not all_symptoms_columns:
        print("Model or encoder not loaded. Attempting to load/train.")
        load_and_train_model()
        if svc_model is None:
            raise RuntimeError("Model could not be loaded or trained.")

    # Normalize input symptoms to match training data format
    processed_symptoms = {}
    for symptom, value in patient_symptoms.items():
        norm_symptom = normalize_symptom_input(symptom)
        if norm_symptom in all_symptoms_columns:
            processed_symptoms[norm_symptom] = 1 if str(value).strip() == '1' else 0
        else:
            print(f"Note: Symptom '{symptom}' not found in trained symptoms. Ignoring.")
    
    # Create input array with all symptoms initialized to 0
    input_features = np.zeros(len(all_symptoms_columns))
    
    # Map provided symptoms to the correct columns
    for i, symptom_col in enumerate(all_symptoms_columns):
        if symptom_col in processed_symptoms and processed_symptoms[symptom_col] == 1:
            input_features[i] = 1

    # Make prediction
    prediction_input = input_features.astype(float).reshape(1, -1)
    predicted_label_encoded = svc_model.predict(prediction_input)
    predicted_disease = label_encoder.inverse_transform(predicted_label_encoded)[0]

    print(f"Predicted disease: {predicted_disease}")
    print("--- Finished get_predicted_value ---")
    return predicted_disease

def get_recommendations(disease: str) -> dict:
    """
    Provides recommendations for a given disease.
    
    Args:
        disease (str): The predicted disease name.
        
    Returns:
        dict: Dictionary containing various recommendations.
    """
    print("--- Starting get_recommendations ---")
    if description_df is None:
        print("Recommendation dataframes not loaded. Attempting to load.")
        load_recommendation_data()
        if description_df is None:
            raise RuntimeError("Recommendation dataframes could not be loaded.")

    recommendations = {
        "description": "",
        "precautions": [],
        "medications": [],
        "workout": [],
        "diet": []
    }

    cleaned_disease = disease.strip().replace('_', ' ').lower()

    # Helper function to get recommendations from a dataframe
    def get_from_df(df, disease_col, value_cols):
        result = []
        row = df[df[disease_col].str.strip().str.replace('_', ' ').str.lower() == cleaned_disease]
        if not row.empty:
            for col in value_cols:
                if col in row.columns and pd.notna(row[col].iloc[0]):
                    if isinstance(row[col].iloc[0], str):
                        result.extend([item.strip() for item in row[col].iloc[0].split(',') if item.strip()])
                    else:
                        result.append(row[col].iloc[0])
        return result

    # Get description
    desc_row = description_df[description_df['Disease'].str.strip().str.replace('_', ' ').str.lower() == cleaned_disease]
    if not desc_row.empty:
        recommendations["description"] = desc_row['Description'].iloc[0]

    # Get other recommendations
    recommendations["precautions"] = get_from_df(precautions_df, 'Disease', 
                                               [f'Precaution_{i}' for i in range(1, 5)])
    recommendations["medications"] = get_from_df(medications_df, 'Disease', 
                                               [f'Medication_{i}' for i in range(1, 5)])
    recommendations["workout"] = get_from_df(workout_df, 'disease', ['workout'])
    recommendations["diet"] = get_from_df(diets_df, 'Disease', 
                                        [f'Diet_{i}' for i in range(1, 5)])

    print("--- Finished get_recommendations ---")
    return recommendations

def get_available_symptoms():
    """Return list of all symptoms the model recognizes"""
    return sorted(all_symptoms_columns)

# Example usage for testing
if __name__ == "__main__":
    load_and_train_model()
    print("Available symptoms:", get_available_symptoms())

    # Test prediction with different symptom formats
    sample_symptoms = {
        'itching': 1,
        'skin rash': 1,  # with space
        'nodal_skin_eruptions': 1,  # with underscore
        'HIGH FEVER': 0,  # uppercase
        'unknown_symptom': 1  # will be ignored
    }

    predicted_disease = get_predicted_value(sample_symptoms)
    print(f"\nPredicted Disease: {predicted_disease}")
    print("Recommendations:", get_recommendations(predicted_disease))