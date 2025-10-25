# Hello Team
This is what I have cooked up as a practice run for the big day tomorrow. I hope we find this useful. REMEMBER WE ARE NOT GOING TO COPY THIS PROJECT AT ALL DURING THE HACKATHON THIS IS SUPER IMPORTANT. THIS IS PURELY FOR PRACTICE PURPOSES

# Muse Brainwave Classifier
A machine learning pipeline to classify motor imagery EEG data (up, down, left, right) collected from the Muse 2 headset.

## 🚀 Overview

1. **Collect Data** — Gather EEG CSV data from the Muse 2 headset.  
2. **Preprocess** — Clean and filter raw signals for noise and artifacts.  
3. **Train Model** — Use a RandomForestClassifier (Scikit-learn) to train on processed features.  
4. **Validate** — Automatically split training data into training and validation sets.  
5. **Predict** — Load the trained model and run inference on new EEG data.

---

## 🧩 Project Structure

/data
├── train_data.csv # Your training data
├── val_data.csv # Automatically generated validation split
/src
├── train_model.py # Trains model and saves it as trained_model.pkl
├── predict.py # Loads model and runs predictions
├── scikit_utils.py # Helper functions (if any)
/Dockerfile
/README.md

yaml
Copy code

---

## 🐳 Run with Docker

### Build the image
```bash
docker build -t muse-classifier .
Start an interactive container
bash
Copy code
docker run -it --rm -v $(pwd):/app muse-classifier bash
🧠 Training the Model
Inside the container:

bash
Copy code
python src/train_model.py
This will:

Load data/train_data.csv

Split it automatically into training (80%) and validation (20%)

Train a Random Forest classifier

Save the model as trained_model.pkl

Export the validation subset to data/val_data.csv

🔍 Validating the Model
After training, you can verify model performance on the held-out validation data:

bash
Copy code
python src/validate_model.py
(If not implemented, you can still inspect accuracy printed at the end of training.)

🔮 Making Predictions
To run inference on EEG data (e.g., the validation split):

bash
Copy code
python src/predict.py
This script:

Loads trained_model.pkl

Reads data/val_data.csv

Outputs predicted class labels and probabilities to the console

⚙️ Notes
Make sure your CSV headers match the training data features (alpha_alpha_power, beta_beta_power, gamma_gamma_power, theta_theta_power, etc.).

Feature names must match exactly for predictions to work.

If you change preprocessing, retrain the model to regenerate the .pkl file.

