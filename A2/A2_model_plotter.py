import seaborn as sns
from matplotlib import pyplot


def plot_history(history):
    """Plot learning curves after the model has been trained"""
    pyplot.title('Model Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train', marker='o', markersize=3)
    pyplot.plot(history.history['val_accuracy'], label='val', marker='o', markersize=3)
    pyplot.legend(loc='upper left')
    pyplot.grid()
    pyplot.show()

    pyplot.title('Model Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.plot(history.history['loss'], label='train', marker='o', markersize=3)
    pyplot.plot(history.history['val_loss'], label='val', marker='o', markersize=3)
    pyplot.legend(loc='upper right')
    pyplot.grid()
    pyplot.show()


def plot_confusion_matrix(dataframe):
    """PLot calculated confusion matrix using seaborn"""
    pyplot.figure(figsize=(8, 8))
    sns.set(font_scale=2)
    sns.heatmap(dataframe, annot=True, cmap=pyplot.cm.Blues)
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.show()