from statistics import mean, stdev
from tools import compress
import numpy as np


def score_predictions(predictions, y_test, odds):

    flatten_predictions = [float(item) for sublist in predictions for item in sublist]
    scores = np.array(y_test - flatten_predictions).tolist()
    excellent_scores = map(lambda elt: True if abs(elt) <= 0.5 else False, scores)
    good_scores = map(lambda elt: True if abs(elt) <= 1.0 and abs(elt) > 0.5 else False, scores)
    bad_scores = map(lambda elt: True if abs(elt) > 1.0 else False, scores)

    simulation_money = 0

    for i in range(len(scores)):
        if flatten_predictions[i] >= 0.5 and y_test[i] == 1.0:
            odd = odds[i, 2].replace(',', '.')
            simulation_money += 2 * float(odd)
        elif flatten_predictions[i] <= -0.5 and y_test[i] == -1.0:
            odd = odds[i, 3].replace(',', '.')
            simulation_money += 2 * float(odd)
        simulation_money -= 2

    teams_predictions = [[team, prediction] for prediction, team in zip(flatten_predictions, odds)]

    excellent_scores_teams = list(compress(teams_predictions, excellent_scores))
    good_scores_teams = list(compress(teams_predictions, good_scores))
    bad_scores_teams = list(compress(teams_predictions, bad_scores))

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
        print("\t\tMin: %.3f" % (min(flatten_predictions)))
        print("\t\tMax: %.3f" % (max(flatten_predictions)))
        print("\t\tMean: %.3f" % (mean(flatten_predictions)))
        print("\t\tStandard deviation: %.3f\n" % (stdev(flatten_predictions)))

    print("The accuracy of the model is: %.1f perc.\n\n" % ((len(excellent_scores_teams) + len(good_scores_teams) / 2) / len(scores) * 100))

    return simulation_money