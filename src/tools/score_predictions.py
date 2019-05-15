from statistics import mean, stdev
from tools.tools import compress
import numpy as np


def score_predictions(predictions, y_test, odds):

    # Compute the score for each sample
    scores = np.array(y_test - predictions).tolist()
    
    # Differentiate the different scores with excellent, good or bad labels
    excellent_scores = map(lambda elt: True if abs(elt) <= 0.5 else False, scores)
    good_scores = map(lambda elt: True if abs(elt) <= 1.0 and abs(elt) > 0.5 else False, scores)
    bad_scores = map(lambda elt: True if abs(elt) > 1.0 else False, scores)

    # Zip the predictions on each game
    teams_predictions = [[team, prediction] for prediction, team in zip(predictions, odds)]

    # Differentiale now on each label we determined before
    excellent_scores_teams = list(compress(teams_predictions, excellent_scores))
    good_scores_teams = list(compress(teams_predictions, good_scores))
    bad_scores_teams = list(compress(teams_predictions, bad_scores))

    # Print the results
    print("The result for each team predicted correctly:\n")
    
    print("\t- with an excellent confidence indice is:\n")
    for score in excellent_scores_teams:
        print("\t\t%s / %s / %.3f / %s - %s" % (score[0][0], score[0][1], score[1], score[0][2], score[0][3]))

    print("\t- with an good confidence indice is:\n")
    for score in good_scores_teams:
        print("\t\t%s / %s / %.3f / %s - %s" % (score[0][0], score[0][1], score[1], score[0][2], score[0][3]))
            
    print("\t- with an bad confidence indice is:\n")
    for score in bad_scores_teams:
        print("\t\t%s / %s / %.3f / %s - %s" % (score[0][0], score[0][1], score[1], score[0][2], score[0][3]))

    if len(predictions) > 1:
        print("\nPredictions stastistics:\n")
        print("\t\tMin: %.3f" % (min(predictions)))
        print("\t\tMax: %.3f" % (max(predictions)))
        print("\t\tMean: %.3f" % (mean(predictions)))
        print("\t\tStandard deviation: %.3f\n" % (stdev(predictions)))

    print("The accuracy of the model is: %.1f perc.\n\n" % ((len(excellent_scores_teams) + len(good_scores_teams) / 2) / len(scores) * 100))


def simulate_bets(predictions, y_test, betting_threshold, odds):
    simulation_money = 0
    low, high = betting_threshold
    for i in range(len(y_test)):
        if predictions[i] >= low and predictions[i] <= high and y_test[i] == 1.0:
            odd = odds[i, 2].replace(',', '.')
            simulation_money += 2 * float(odd)
        elif predictions[i] <= -low and predictions[i] >= -high and y_test[i] == -1.0:
            odd = odds[i, 3].replace(',', '.')
            simulation_money += 2 * float(odd)
        if (predictions[i] >= low and predictions[i] <= high) or (predictions[i] <= -low and predictions[i] >= -high):
            simulation_money -= 2

    return simulation_money