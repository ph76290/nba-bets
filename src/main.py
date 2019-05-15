from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(3)
from matplotlib import pyplot as plt
from tools.score_predictions import score_predictions, simulate_bets
from fit_and_predict import fit_model, predict_teams_result
from tools.split_dataset import split_dataset
from data_processing import build_train_test_data
from scraping.odds_scraping import read_odds
from tools.tools import df_describe
from tools.plot import plot_model_history, plot_money_earned
import os


# Hyper-parameters
chunks = 3
epoch = 30
dropout = 0.4
extra_hidden_layer = True
betting_threshold = (0.6, 0.8)
seasons = range(2016, 2020)
train_proportion = 0.7
validation_proportion = 0.2
# Visualize data and results
describe_df = False
plot_money = True
plot_history = True
print_scores = False


model = None
total_money = 0
all_money = []
all_history = []

# Iterate over the seasons
for season in seasons:
    
    # Path to datasets
    path = "../data/season_" + str(season) + "/teams"

    # Build dataset
    df = build_train_test_data(path)
    df_without_na = df.dropna()
    if describe_df:
        df_describe(df_without_na)
    length_odds = len(df) - len(df_without_na)

    values = df_without_na.values
    chunk = int(len(values) / chunks)

    # Fetch odds
    odds = read_odds(season, df)[length_odds:]

    # Iterate over the different parts of the season
    for i in range(chunks):

        # Split datasets and odds
        nb_train_samples = int(chunk * train_proportion)
        nb_validation_samples = int(chunk * validation_proportion)
        sets = split_dataset(nb_train_samples, nb_validation_samples, values[i * chunk:(i + 1) * chunk])
        odds_values = odds.values[i * chunk + nb_train_samples + nb_validation_samples:(i + 1) * chunk, :]

        # Fitting the data
        model, history = fit_model(model, sets, epoch, extra_hidden_layer, dropout)

        # Predict teams results
        print("Predicting the results for the part {}/{} of the season {}...\n".format(i + 1, chunks, season))
        predictions = predict_teams_result(model, sets[2][0])

        # Print the score and predictions for each test samples
        if print_scores:
            score_predictions(predictions, sets[2][1], odds_values)

        # Compute the money we could earn for this part of this season
        simulation_money = simulate_bets(predictions, sets[2][1], betting_threshold, odds_values)

        all_money.append(simulation_money)
        all_history.append(history)
        total_money += simulation_money

# Plot the validation and training loss over epochs
if plot_history:
    plot_model_history(all_history, seasons, chunks, dropout, extra_hidden_layer, epoch)

print("You would earn {}".format(total_money))

# Plot the money earn over each part of each season
if plot_money:
    plot_money_earned(all_money, betting_threshold, seasons, chunks)