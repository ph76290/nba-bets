import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from build_dataset_tools.load_file import load_csv_into_dataframe
from encode_features import encode_location, encode_streak, encode_overtimes, encode_result, encode_start_time, encode_face_to_face, encode_last_n
from datetime import datetime
import os
from tools import min_max_normalization, swap_game, get_teams, element_in_string, get_team_from_full_name, get_all_columns


def build_train_test_data(path):
    if os.path.exists(path + 'data_processed.csv'):
        return pd.read_csv(path + 'data_processed.csv', index_col=[0, 1])
    df = pd.DataFrame()
    if os.path.isfile(path):
        df = build_train_test_data_single_file(path)
    else:
        for paths, subdirs, files in os.walk(path):
            if paths.split('/')[-1] == "teams":
                for file in files:
                    new_df = build_train_test_data_single_file(os.path.join(paths, file))
                    if df.shape == (0, 0):
                        df = new_df
                    else:
                        df = df.merge(
                            new_df,
                            left_index=True,
                            right_index=True,
                            how='outer',
                            on=get_all_columns()
                        )

        to_delete = []
        # Merge the same games
        for i in range(df.shape[0]):
            first_index = df.index[i]
            second_index = (first_index[0], swap_game(first_index[1]))

            home_index = first_index if df.iloc[i]['game_location'] == 0.0 else second_index
            away_index = first_index if df.iloc[i]['game_location'] == 1.0 else second_index

            if not(away_index in to_delete):
                to_delete.append(away_index)

                df.loc[home_index]['face_to_face'] -= df.loc[away_index]['face_to_face']
                df.loc[home_index]['game_start_time'] -= df.loc[away_index]['game_start_time']                    
                df.loc[home_index]['game_streak'] -= df.loc[away_index]['game_streak']
                df.loc[home_index]['pts'] -= df.loc[away_index]['pts']
                df.loc[home_index]['opp_pts'] -= df.loc[away_index]['opp_pts']
                df.loc[home_index]['overtimes'] -= df.loc[away_index]['overtimes']
                df.loc[home_index]['time_between_games'] -= df.loc[away_index]['time_between_games']
                df.loc[home_index]['game_result'] -= df.loc[away_index]['game_result']
                df.loc[home_index]['ast'] -= df.loc[away_index]['ast']
                df.loc[home_index]['tov'] -= df.loc[away_index]['tov']                    
                df.loc[home_index]['fg'] -= df.loc[away_index]['fg']
                df.loc[home_index]['fga'] -= df.loc[away_index]['fga']
                df.loc[home_index]['fg3'] -= df.loc[away_index]['fg3']
                df.loc[home_index]['fg3a'] -= df.loc[away_index]['fg3a']
                df.loc[home_index]['ft'] -= df.loc[away_index]['ft']
                df.loc[home_index]['fta'] -= df.loc[away_index]['fta']
                df.loc[home_index]['orb'] -= df.loc[away_index]['orb']
                df.loc[home_index]['drb'] -= df.loc[away_index]['drb']                    
                df.loc[home_index]['stl'] -= df.loc[away_index]['stl']
                df.loc[home_index]['blk'] -= df.loc[away_index]['blk']
                df.loc[home_index]['pf'] -= df.loc[away_index]['pf']
                df.loc[home_index]['plus_minus'] -= df.loc[away_index]['plus_minus']
                for team in get_teams():
                    df.loc[home_index][team] -= df.loc[away_index][team]

            i += 1

    df.drop(to_delete, axis=0, inplace=True)
    df.drop(columns=['game_location', 'game_start_time'], axis=1, inplace=True)

    # Min Max normalization
    df["game_streak"] = min_max_normalization(df["game_streak"])
    df["pts"] = min_max_normalization(df["pts"])
    df["opp_pts"] = min_max_normalization(df["opp_pts"])
    df["overtimes"] = min_max_normalization(df["overtimes"])
    df["time_between_games"] = min_max_normalization(df["time_between_games"])
    df['ast'] = min_max_normalization(df['ast'])
    df['tov'] = min_max_normalization(df['tov'])
    df['fg'] = min_max_normalization(df['fg'])
    df['fga'] = min_max_normalization(df['fga'])
    df['fg3'] = min_max_normalization(df['fg3'])
    df['fg3a'] = min_max_normalization(df['fg3a'])
    df['ft'] = min_max_normalization(df['ft'])
    df['fta'] = min_max_normalization(df['fta'])
    df['orb'] = min_max_normalization(df['orb'])
    df['drb'] = min_max_normalization(df['drb'])
    df['stl'] = min_max_normalization(df['stl'])
    df['blk'] = min_max_normalization(df['blk'])
    df['pf'] = min_max_normalization(df['pf'])
    df['plus_minus'] = min_max_normalization(df['plus_minus'])

    df.to_csv(path + 'data_processed.csv', sep=',', encoding='utf-8')
    return df


def build_train_test_data_single_file(filename):
    df = load_csv_into_dataframe(filename, [0])

    # Process columns
    df['date_game'] = pd.to_datetime(df['date_game'], format='%a, %b %d, %Y')

    # The result of the game
    df['game_result'] = df['game_result'].apply(encode_result).astype("float32")
    
    # Current team
    for team in get_teams():
        if team == element_in_string(get_teams(), filename):
            df[team] = np.ones(df.shape[0]).astype('float32')
        else:
            df[team] = np.zeros(df.shape[0]).astype('float32')  

    # Game start time
    df["game_start_time"] = df["game_start_time"].apply(encode_start_time).astype("float32")
    # Location: Home 1 / Away 0
    df["game_location"] = df["game_location"].apply(encode_location).astype("float32")
    # Last 3 games average points scored
    df['pts'] = encode_last_n(df['pts'], 3).astype("float32")
    # Streak of wins or losses
    df['game_streak'] = df['game_streak'].apply(encode_streak).shift(1).astype("float32")
    # Win ratio
    df['win_ratio'] = (df['wins'] / (df['wins'] + df['losses'])).shift(1).astype("float32")
    # Last game number of overtimes
    df["overtimes"] = df["overtimes"].astype("str").apply(encode_overtimes).shift(1).astype("float32")
    # Last 3 games average points opponent scored
    df['opp_pts'] = encode_last_n(df['opp_pts'], 3).astype("float32")
    # Time between the last game and the current one
    df['time_between_games'] = ((df.date_game - df.date_game.shift(1)).astype('timedelta64[h]') / 24).astype("float32")
    # Face to face ratio between teams
    df['face_to_face'] = encode_face_to_face(df).astype('float32')

    # Last 3 games assists average
    df['ast'] = encode_last_n(df['ast'], 3)
    # Last 3 games turnovers average
    df['tov'] = encode_last_n(df['tov'], 3)
    # Last 3 games field goals made average
    df['fg'] = encode_last_n(df['fg'], 3)
    # Last 3 games field goal attempts average
    df['fga'] = encode_last_n(df['fga'], 3)
    # Last 3 games 3-pointers made average
    df['fg3'] = encode_last_n(df['fg3'], 3)
    # Last 3 games 3-pointer attempts average
    df['fg3a'] = encode_last_n(df['fg3a'], 3)
    # Last 3 games free throws made average
    df['ft'] = encode_last_n(df['ft'], 3)
    # Last 3 games free throw attempts average
    df['fta'] = encode_last_n(df['fta'], 3)
    # Last 3 games offensive rebounds average
    df['orb'] = encode_last_n(df['orb'], 3)
    # Last 3 games defensive rebounds average
    df['drb'] = encode_last_n(df['drb'], 3)
    # Last 3 games steals average
    df['stl'] = encode_last_n(df['stl'], 3)
    # Last 3 games blocks average
    df['blk'] = encode_last_n(df['blk'], 3)
    # Last 3 games fouls committed average
    df['pf'] = encode_last_n(df['pf'], 3)
    # Last 3 games total plus_minus average
    df['plus_minus'] = encode_last_n(df['plus_minus'], 3)
    #datetimes = list(df.index)
    team_name = element_in_string(get_teams(), filename)
    df['games'] = np.array([team_name + "-" + get_team_from_full_name(opp_name) for opp_name in df['opp_name'].values])
    # Set datetime as index of the DataFrame
    df.set_index(["date_game", "games"], inplace=True)

    # Access only some columns
    df = df.loc[:, get_all_columns()]

    return df