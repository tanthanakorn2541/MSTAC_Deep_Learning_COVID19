from Modeling.DataHandler import DataHandler
from Modeling.ModelCreator import ModelCreator
from Modeling.PerformanceEvaluator import PerformanceEvaluator

def run_model(model_type, labels, epochs=20, batch_size=32):
    train_data_dir = '../Dataset_holdout/train'
    data_handler = DataHandler(train_data_dir)

    X_train, X_val, y_train, y_val = data_handler.load_training_data(model_type)

    model_creator = ModelCreator()
    model = model_creator.create_model(model_type)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    y_pred = model.predict_classes(X_val)
    evaluator = PerformanceEvaluator(y_val, y_pred, labels)
    evaluator.plot_confusion_matrix()
    evaluator.print_classification_report()
    
    # Save the model
    model.save(f'saved_models/model{model_type}/CNN')

def save_model():
    run_model(1, ['Unhealthy', 'Healthy'])
    run_model(2, ['COVID-19', 'Non-COVID-19'])

def main():
    save_model()
    
if __name__ == "__main__":
    main()
