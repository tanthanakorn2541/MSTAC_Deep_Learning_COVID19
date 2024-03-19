from Modeling.DataHandler import DataHandler
from Modeling.PerformanceEvaluator import PerformanceEvaluator
from train_and_evaluate import evaluate_and_combine_predictions, PerformanceEvaluator
from tensorflow.keras.models import load_model
import numpy as np

def load_and_predict_model(model_path, X_test):
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return y_pred_classes.tolist()

def preprocess_multiclass_predictions(y_pred_classes):
    position = [i for i, pred_class in enumerate(y_pred_classes) if pred_class == 1]
    y_pred_multiclass = [2] * len(position)
    return position, y_pred_multiclass

def split_multiclass_samples(X_test, y_test, position):
    X_test_multiclass = [X_test.pop(i) for i in reversed(position)]
    y_true_multiclass = [y_test.pop(i) for i in reversed(position)]
    return X_test_multiclass, y_true_multiclass

def evaluate_and_combine_predictions(Model_healthy, Model_COVID, X_test, y_test):
    y_pred_healthy = load_and_predict_model(Model_healthy, X_test)
    position, y_pred_multiclass = preprocess_multiclass_predictions(y_pred_healthy)

    X_test_multiclass, y_true_multiclass = split_multiclass_samples(X_test, y_test, position)

    y_pred_COVID = load_and_predict_model(Model_COVID, X_test)
    y_pred_COVID = np.argmax(y_pred_COVID, axis=1).tolist()

    y_true_combined = y_true_multiclass + y_test
    y_pred_combined = y_pred_multiclass + y_pred_COVID

    return y_true_combined, y_pred_combined

def main():
    test_data_dir = '../Dataset_holdout/test'
    data_handler = DataHandler(test_data_dir)
    X_test, y_test = data_handler.load_testing_data(model_type=3)
    
    Model_healthy_path = '../saved_models/model1/CNN'
    Model_COVID_path = '../saved_models/model2/CNN'

    y_true, y_pred = evaluate_and_combine_predictions(Model_healthy_path, Model_COVID_path, X_test, y_test)
    y_TRUE = np.argmax(y_true, axis=1)
    
    evaluator = PerformanceEvaluator(y_TRUE, y_pred, ["COVID-19", "Non-COVID-19", "Healthy"])
    evaluator.plot_confusion_matrix()
    evaluator.print_classification_report()

if __name__ == "__main__":
    main()
