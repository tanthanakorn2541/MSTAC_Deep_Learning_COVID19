# MSTAC: A Multi-Stage Automated Classification of COVID-19 Chest X-ray Images Using Stacked CNN Models

This project focuses on the classification of COVID-19 cases using the stacking two deep learning models. The project is developed in Python and utilizes libraries such as TensorFlow, scikit-learn, and others for data processing, model creation, evaluation, and model saving. MSTAC employs two classification stages: the first distinguishes healthy from unhealthy cases, and the second further classifies COVID-19 and non-COVID-19 cases.

## Project Structure

The project is structured as follows:

- **COVID_Classification_Project/**
  - **Modeling/**
    - **DataHandler.py:** Handles data loading and preprocessing.
    - **ModelCreator.py:** Creates machine learning and deep learning models.
    - **PerformanceEvaluator.py:** Evaluates model performance and generates reports.
  - **Main/**
    - **train_and_evaluate.py:** Runs the training, and saveing processes.
    - **Main.py:** Runs the stacking model, evaluation, and prediction processes.

## Running the Project

1. Run `train_and_evaluate.py` to train and evaluate the deep learning models.
2. Run `main.py` to perform predictions using the stacked models.

## Libraries Required

Ensure you have the following libraries installed to run the project:

- TensorFlow
- scikit-learn
- numpy
- matplotlib
- seaborn (for visualization)
- pandas (if needed for data handling)

## Usage

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Navigate to the `COVID_Classification_Project/Main` directory.
4. Run `python train_and_evaluate.py` to train and evaluate the models.
5. Run `python main.py` for predictions.

## Notes

- Make sure to set the appropriate paths for data and model saving/loading in the code files.
- Adjust hyperparameters and model architectures as needed for optimal performance.
