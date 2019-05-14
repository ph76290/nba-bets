from tools.tools import get_odds

def split_dataset(nb_train_samples, nb_validation_samples, values, season, length_odds):

    odds = get_odds(season)
    odds['winner_rate'] = odds.apply(lambda row: row['odd_home_team'] if row['results'] == 1.0 else row['odd_away_team'], axis=1)
    odds = list(map(lambda x: float(x.replace(',', '.')), odds.values[:, 5]))
    odds = odds[length_odds:]

    # Split the dataset into train and test sets
    odds_train = odds[:nb_train_samples]
    odds_validation = odds[nb_train_samples:nb_validation_samples + nb_train_samples]
    train = values[:nb_train_samples, :]
    validation = values[nb_train_samples:nb_validation_samples + nb_train_samples, :]
    test = values[nb_validation_samples + nb_train_samples:, :]

    # Split the train and test sets into inputs and outputs
    x_train, y_train = train[:, :-1], train[:, -1] * odds_train
    x_validation, y_validation = validation[:, :-1], validation[:, -1] * odds_validation
    x_test, y_test = test[:, :-1], test[:, -1]

    # Reshape the sets
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_validation = x_validation.reshape((x_validation.shape[0], 1, x_validation.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    return ((x_train, y_train), (x_validation, y_validation), (x_test, y_test))
