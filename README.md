# deep_learning_challenge
## Alphabet Soup Charity Deep Learning Challenge

## Overview

The goal of this project is to develop a binary classification model using deep learning to predict whether organizations funded by Alphabet Soup will be successful. By leveraging machine learning and neural networks, the model aims to assist Alphabet Soup in selecting applicants with the best chances of success.

## Dataset

The dataset provided contains information on over 34,000 organizations that received funding from Alphabet Soup. It includes various categorical and numerical variables that describe each organization's characteristics and funding details.

## Files in Repository

AlphabetSoupCharity.ipynb: Jupyter Notebook for data preprocessing, model training, and evaluation.

AlphabetSoupCharity_Optimization.ipynb: Jupyter Notebook for optimizing the model to improve accuracy.

AlphabetSoupCharity.h5: Saved model file from the initial deep learning model.

AlphabetSoupCharity_Optimization.h5: Saved model file from the optimized deep learning model.

charity_data.csv: Input dataset used for training and evaluation.

README.md: Project documentation.

## Data Preprocessing

Target Variable: IS_SUCCESSFUL (indicates whether the funding led to a successful outcome).

Feature Variables: All columns except EIN and NAME, which are identifiers and not useful for prediction.

Dropped Variables: EIN, NAME.

Categorical Encoding: One-hot encoding was applied to categorical features.

Feature Scaling: StandardScaler() was used to normalize numerical features.

Data Splitting: The dataset was divided into training and testing sets using train_test_split().

Model Architecture

Input Features: Number of input features based on the processed dataset.

Hidden Layers:

Layer 1: Dense layer with ReLU activation function.

Layer 2: Additional Dense layer with ReLU activation function.

Output Layer: Single neuron with a sigmoid activation function for binary classification.

Compilation:

Loss Function: Binary cross-entropy

Optimizer: Adam

## Training:

Number of Epochs: Initially 100, adjusted during optimization.

Model checkpointing to save weights every five epochs.

Model Performance

Initial Model Accuracy: Approximately below 75%.

## Optimization Attempts:

Adjusted categorical binning and feature selection. such as 

the value of layer has been changed like first layer =50, second layer= 100,  Third layer= 200, fourth layer = 250, fifth layer = 300.

Increased neurons in hidden layers.

Increased the number of hidden layers, we put 1 to 5 hidden layers.

Experimented with different activation functions.

Adjusted the number of training epochs 200.

Optimized Model Accuracy: Attempted to surpass 73% may vary could not get 75% .

## Summary & Recommendations

# Findings:

The deep learning model provides a reasonable accuracy but may be improved further.

Additional feature engineering and hyperparameter tuning could help boost performance.

## Alternative Approach:

Consider using traditional machine learning classifiers such as Random Forest, Gradient Boosting, or XGBoost.

Feature selection techniques like PCA or Recursive Feature Elimination (RFE) may improve performance.

A different deep learning architecture with dropout layers or batch normalization may enhance accuracy.

## How to Run the Project

Clone this repository.

Open AlphabetSoupCharity.ipynb in Google Colab or Jupyter Notebook.

Run all cells to preprocess the data, train the model, and evaluate performance.

To run the optimized model, open and execute AlphabetSoupCharity_Optimization.ipynb.

Review the saved models (.h5 files) for further analysis.

## Technologies Used

Python

Pandas, NumPy

TensorFlow, Keras

Scikit-learn

Matplotlib, Seaborn


