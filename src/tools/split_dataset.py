def split_dataset(nb_train_samples, nb_validation_samples, values):

    # Split the dataset into train and test sets
    train = values[:nb_train_samples, :]
    validation = values[nb_train_samples:nb_validation_samples + nb_train_samples, :]
    test = values[nb_validation_samples + nb_train_samples:, :]

    # Split the train and test sets into inputs and outputs
    x_train, y_train = train[:, :-1], train[:, -1]
    x_validation, y_validation = validation[:, :-1], validation[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    # Reshape the sets
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_validation = x_validation.reshape((x_validation.shape[0], 1, x_validation.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    return ((x_train, y_train), (x_validation, y_validation), (x_test, y_test))
