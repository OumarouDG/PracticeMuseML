# Hello Team
This is what I have cooked up as a practice run for the big day tomorrow. I hope we find this useful. REMEMBER WE ARE NOT GOING TO COPY THIS PROJECT AT ALL DURING THE HACKATHON THIS IS SUPER IMPORTANT. THIS IS PURELY FOR PRACTICE PURPOSES

# Muse Brainwave Classifier
A machine learning pipeline to classify motor imagery EEG data (up, down, left, right) collected from the Muse 2 headset.

## Overview
1. Collect EEG data via Muse 2 (CSV format)
2. Preprocess and clean (artifact removal, filtering)
3. Train model using Scikit-learn
4. Save model as `.pkl` for later inference
5. Run predictions in real-time or from stored EEG data

## Run Locally with Docker

```bash
# Build the image
docker build -t muse-classifier .

# Start container
docker run -it --rm -v $(pwd):/app muse-classifier bash

# Inside container
python src/train_model.py
