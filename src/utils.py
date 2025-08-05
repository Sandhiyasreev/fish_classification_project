import matplotlib.pyplot as plt

def plot_history(history):
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title("Accuracy")
    plt.show()
