import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class PerformanceEvaluator:
    def __init__(self, y_true, y_pred, classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def print_classification_report(self):
        report = classification_report(self.y_true, self.y_pred)
        print(report)