from matplotlib import pyplot as plt

def plot_model_history(history, seasons, chunks, dropout, extra_hidden_layer, epoch):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Dropout: {} / Extra hidden layer: {} / Epoch: {}".format(dropout, extra_hidden_layer, epoch))

    for i in range(len(history)):

        plt.subplot(len(seasons), chunks, i + 1)
        plt.plot(history[i].history['loss'], label='train')
        plt.plot(history[i].history['val_loss'], label='test')
        plt.legend(prop={'size': 6})

    plt.show()


def plot_money_earned(money, betting_threshold, seasons, chunks):
    plt.plot(money)
    plt.title("Range for betting: {} / Seasons: {} / Chunks: {}".format(betting_threshold, seasons, chunks))
    plt.show()