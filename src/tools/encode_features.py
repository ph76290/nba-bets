import numpy as np
import pandas as pd
from tools import get_teams_full_name

def encode_streak(elt):
    if (elt[0] == 'L'):
        return -1 * int(elt[-1])
    return int(elt[-1])

def encode_team(elt):
    array = np.zeros(30)
    team_pos = get_teams_full_name().index(elt)
    array[team_pos] = 1
    return array

def encode_location(elt):
    if elt == '@':
        return 1
    return 0

def encode_overtimes(elt):
    if not(elt):
        return 0
    if not(elt.startswith("OT")):
        return int(elt[0])
    return 1

def encode_result(elt):
    if elt == "L":
        return 0
    return 1

def encode_start_time(elt):
    encoded = 10
    elt_splited = elt.split(':')
    encoded *= int(elt_splited[0])
    if elt_splited[1][0] == '3':
        encoded += 5
    if elt_splited[1][-1] == 'p':
        encoded += 120
    return encoded

def encode_face_to_face(df):
    df['face_to_face'] = np.zeros(df.shape[0])
    for i in range(1, df.shape[0]):
        new_var = df[:i]
        current_team = df.iloc[i]['opp_name']
        new_df = new_var[new_var['opp_name'] == current_team]
        nb_values = new_df['game_result'].value_counts()
        nb_wins = nb_values.index[0] if len(nb_values.index) else 0.5
        nb_games = new_df.shape[0] if new_df.shape[0] != 0 else 1
        df.ix[i, 'face_to_face'] = nb_wins / nb_games
    return df['face_to_face']

def encode_last_n(serie, n):
    lastN_sum = 0
    new_serie = pd.Series(data=np.zeros(serie.shape[0]))
    new_serie[0] = np.nan
    for i in range(1, serie.shape[0]):
        lastN_sum += serie[i - 1]
        if i >= n:
            new_serie[i] = lastN_sum / n
            lastN_sum -= serie[i - n]
        else:
            new_serie[i] = lastN_sum / i
    return new_serie