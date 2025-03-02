# Gantry_scanner
The gantry scanner based on gimbal is a hardware setup that can be used to acquire the ultasound images of excised tissue or organ for better analysis. This assist the pathologist to improve their accuracy in tumor analysis by providing the location from where they have to cut the tissue for analysis. This also helps the researchers to acquire the ultasound images of different tissue/organ and use these dataset to solve various research problems.

Low cost gimabal based ganrty:
For createting the gimbal based setup we have design and implemented the multiple 3D model includeing the enclosure and gimbal based gantry header that can hold the ultrasound transducer and move in differenct angles for acquiring and predicting the probability of dieseased tissue/organ.

Process:
To execute the code for testing first we acqurire the ultrasound images using gimbal basaed gantry. Then train these images to predict the optimal tumor location and probability of disease using deep learning model. Then we integrate the model in out setup to predict the tumor location based on our algorithm.

This project requires the following dependencies to be installed before running the code.

## Prerequisites

Ensure you have Python installed. You can check your Python version with:

python --version

To install the required dependencies, run:

pip install -r requirements.txt

## Workflow Overview
Follow these steps to execute the full pipeline:

1. Data Preparation
Run the train_val_split_h02.py script to split the dataset into training and validation sets.
python train_val_split_h02.py
2. Model Training
Train the deep learning model using the train.py script.
python train.py
This script will:

Load the dataset.
Train the model.
Save the trained model for further use.

3. Gantry Integration & Tissue Localization
Once the model is trained, integrate it with the automated ultrasound gantry by running localization3.py.
python localization3.py
This script will:

Load the trained model.
Execute localization to determine the optimal tissue slice.
Control the angular ultrasound gantry by integrating both the Python-based and Arduino-based implementations.

Additional Notes
Ensure that the Arduino is properly connected before running localization03.py.
The trained model should be available in the specified directory before executing the localization script.
