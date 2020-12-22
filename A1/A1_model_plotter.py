import glob
import os
import seaborn as sns
from matplotlib import pyplot
from A1_file_utils import load_object

"""
Plot learning curves from history
Display the confusion matrix in a heat map 
"""

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


def plot_graph(title, x_label, y_label, data_item):
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    for file in glob.glob('history_*.data'):
        history = load_object(file)
        legend = os.path.basename(file).replace('history_', '').replace('.data', '')
        pyplot.plot(history[data_item], label=legend, marker='o', markersize=3)
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


def plot_file_history():
    """Plot learning curves from saved data"""
    plot_graph('Model Training Loss', 'Epoch', 'Loss', 'loss')
    plot_graph('Model Validation Loss', 'Epoch', 'Loss', 'val_loss')
    plot_graph('Model Training Accuracy', 'Epoch', 'Accuracy', 'accuracy')
    plot_graph('Model Validation Accuracy', 'Epoch', 'Accuracy', 'val_accuracy')


if __name__ == '__main__':
    # plot_file_history()
    pyplot.title('Training Time vs Batch Size')
    pyplot.xlabel('Batch Size')
    pyplot.ylabel('Training Time (s)')
    X = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    Y = [1401, 761, 533, 463, 455, 396, 376, 363, 346, 370, 468]
    pyplot.plot(X, Y, marker='o', markersize=3)
    pyplot.xscale('log')
    pyplot.grid()
    pyplot.show()
