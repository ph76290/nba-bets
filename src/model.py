from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tools.tools import get_odds

def lstm_model(shape, season, dropout=0.4, extra_hidden_layer=True, neurons_on_first_layer=32):
    model = Sequential()
    model.add(LSTM(neurons_on_first_layer, input_shape=(shape[1], shape[2])))
    if dropout != 0.0:
        model.add(Dropout(dropout))
    if extra_hidden_layer:
        model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss=custom_loss_wrapper(season), optimizer='adam', metrics=["accuracy"])
    return model


def custom_loss_wrapper(season):
    odds = get_odds(season)
    odds['winner_rate'] = odds.apply(lambda row: row['odd_home_team'] if row['results'] == 1.0 else row['odd_away_team'], axis=1)
    print(odds)
    odds = list(map(lambda x: float(x.replace(',', '.')), odds.values[:, 5]))
    def custom_loss(y_true, y_pred):
        #print(y_true, y_pred)
        loss = (y_true - y_pred) * odds
        print(loss)
        return loss * loss
    return custom_loss
