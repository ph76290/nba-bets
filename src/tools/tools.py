import pandas as pd

def element_in_string(l, s: str):
    for team in l:
        if team in s:
            return team
    return None

def get_all_columns():
    columns = [
            "game_location",
            "game_start_time",
            "face_to_face",
            "pts",
            "opp_pts",
            "game_streak",
            "win_ratio",
            "overtimes",
            "time_between_games",
            "3ast",
            "3tov",
            "3fg",
            "3fga",
            "3fg3",
            "3fg3a",
            "3ft",
            "3fta",
            "3orb",
            "3drb",
            "3stl",
            "3blk",
            "3pf",
            "3plus_minus",
            "8ast",
            "8tov",
            "8fg",
            "8fga",
            "8fg3",
            "8fg3a",
            "8ft",
            "8fta",
            "8orb",
            "8drb",
            "8stl",
            "8blk",
            "8pf",
            "8plus_minus"
        ] + get_teams()
    columns.append("game_result")
    return columns

def get_teams():
    teams = ['BRK', 'PHI', 'MIL', 'CHI', 'CLE', 'BOS', 'LAC', 'MEM', 'ATL', 'MIA', 'CHO', 'UTA', 'SAC', 'NYK', 'LAL', 'ORL', 'DAL', 'DEN', 'IND', 'NOP', 'DET', 'TOR', 'HOU', 'SAS', 'PHO', 'OKC', 'MIN', 'POR', 'GSW', 'WAS']
    return teams

def get_teams_full_name():
    long_name_teams = ["Brooklyn Nets", "Philadelphia 76ers", "Milwaukee Bucks", "Chicago Bulls", "Cleveland Cavaliers", "Boston Celtics", "Los Angeles Clippers", "Memphis Grizzlies", "Atlanta Hawks", "Miami Heat", "Charlotte Hornets", "Utah Jazz", "Sacramento Kings", "New York Knicks", "Los Angeles Lakers", "Orlando Magic", "Dallas Mavericks", "Denver Nuggets", "Indiana Pacers", "New Orleans Pelicans", "Detroit Pistons", "Toronto Raptors", "Houston Rockets", "San Antonio Spurs", "Phoenix Suns", "Oklahoma City Thunder", "Minnesota Timberwolves", "Portland Trail Blazers", "Golden State Warriors", "Washington Wizards"]
    return long_name_teams

def get_team_from_full_name(name):
    full_names = get_teams_full_name()
    for i in range(len(full_names)):
        if name in full_names[i]:
            return get_teams()[i]
        if "Timber" in name:
            return "MIN"
    return None

def get_team_from_short_name(name):
    short_names = get_teams()
    for i in range(len(short_names)):
        if name in short_names[i]:
            return get_teams_full_name()[i]
    return None

def min_max_normalization(serie):
    mi = serie.min()
    ma = serie.max()
    return (serie - mi) / (ma - mi)

def df_describe(df):
    print(df.info())
    print(df.shape)
    print(df.describe())
    print(df.head(5))
    print(df.tail(5))


def compress(l1, l2):
    return [x for x, y in zip(l1, l2) if y]


def swap_game(s):
    new_s = s.split('-')
    return new_s[1] + '-' + new_s[0]

def get_odds(season):
    odds = pd.read_csv("../data/season_" + str(season) + "/odds/odds.csv")
    results = pd.read_csv("../data/season_" + str(season) + "/teams/data_processed.csv")['game_result']
    odds['results'] = results
    return odds