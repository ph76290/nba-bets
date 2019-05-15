from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from model import lstm_model


def fit_model(model, sets, epoch, extra_hidden_layer, dropout):

    # Deconstruct the sets object to get every sets (features + labels)
    ((x_train, y_train), (x_validation, y_validation), (x_test, y_test)) = sets

    # If the model doesn't exist yet, create the lstm model
    if model == None:
        model = lstm_model(x_train.shape, dropout=dropout, extra_hidden_layer=extra_hidden_layer)

    # Fit the model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epoch,
        shuffle=False,
        verbose=0,
    )

    return model, history


def predict_teams_result(model, x_test):

    # Predict on the test set
    predictions = model.predict(x_test)

    # Flatten the predictions
    flatten_predictions = [float(item) for sublist in predictions for item in sublist]

    return flatten_predictions