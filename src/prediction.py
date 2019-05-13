from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from model import lstm_model
from tools.score_predictions import score_predictions
from matplotlib import pyplot as plt



def predict_teams_result(sets, model, odds, season, epoch, extra_hidden_layer, dropout):

    ((x_train, y_train), (x_validation, y_validation), (x_test, y_test)) = sets

    if model == None:
        model = lstm_model(x_train.shape, season, dropout=dropout, extra_hidden_layer=extra_hidden_layer)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epoch,
        shuffle=False,
        verbose=0,
    )

    predictions = model.predict(x_test)
    simulation_money = score_predictions(predictions, y_test, odds)
    
    plt.show()

    return model, simulation_money, history