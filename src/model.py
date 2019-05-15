from keras.models import Sequential
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from tools.tools import get_odds

def lstm_model(shape, dropout=0.4, extra_hidden_layer=True, neurons_on_first_layer=32):
    model = Sequential()
    model.add(LSTM(neurons_on_first_layer, input_shape=(shape[1], shape[2])))
    if dropout != 0.0:
        model.add(Dropout(dropout))
    if extra_hidden_layer:
        model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mae', optimizer='adam', metrics=["accuracy"])
    return model


def custom_loss(y_true, y_pred):
    signs = K.tf.sign(y_true)
    loss = (signs - y_pred) * y_true
    print(loss)
    return loss * loss
