from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(3)
from matplotlib import pyplot as plt
from prediction import predict_teams_result
from tools.split_dataset import split_dataset
from data_processing import build_train_test_data
from scraping.odds_scraping import read_odds
from tools.tools import df_describe
import os


# Hyper-parameters
chunks = 3
epoch = 30
dropout = 0.4
betting_threshold = (0.6, 0.8)
extra_hidden_layer = True
seasons = range(2016, 2020)


model = None
total_money = 0
plot_money = []
plot_history = []


for season in seasons:
    
    path = "../data/season_" + str(season) + "/teams"

    # Build dataset
    df = build_train_test_data(path)
    df_without_na = df.dropna()
    length_odds = len(df) - len(df_without_na)

    #df_describe(df_without_na)
    values = df_without_na.values
    chunk = int(len(values) / chunks)

    # Fetch odds
    odds = read_odds(season, df)[length_odds:]

    for i in range(chunks):
        nb_train_samples = int(chunk * 0.7)
        nb_validation_samples = int(chunk * 0.2)

        sets = split_dataset(nb_train_samples, nb_validation_samples, values[i * chunk:(i + 1) * chunk], season, length_odds)
        odds_values = odds.values[i * chunk + nb_train_samples + nb_validation_samples:(i + 1) * chunk, :]

        # Predict teams results
        print("Predicting the results for the part {}/{} of the season {}...\n".format(i + 1, chunks, season))
        model, simulation_money, history = predict_teams_result(sets, model, odds_values, season, epoch, extra_hidden_layer, dropout, betting_threshold)

        plot_money.append(simulation_money)
        plot_history.append(history)

        total_money += simulation_money


fig = plt.figure(figsize=(10, 10))
fig.suptitle("dropout: {} / extra hidden layer: {} / epoch: {}".format(dropout, extra_hidden_layer, epoch))

for i in range(len(plot_history)):

    plt.subplot(len(seasons), chunks, i + 1)
    plt.plot(plot_history[i].history['loss'], label='train')
    plt.plot(plot_history[i].history['val_loss'], label='test')
    plt.legend(prop={'size': 6})

plt.show()

print("You would earn {}".format(total_money))

plt.plot(plot_money)
plt.show()
