from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """
    Prints classification report.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """
    Plots confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()