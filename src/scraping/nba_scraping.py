from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from tools.tools import element_in_string, get_teams, get_team_from_full_name
from datetime import datetime
import pandas as pd
import numpy as np
import os


url_base = 'https://www.basketball-reference.com/'
url_base_team = url_base + "teams/"
url_season_stats = '_games.html'


def fetch_all_stats(seasons, teams, rewrite=False, write_game=False):

    for season in range(seasons[0], seasons[1]):
        if not os.path.exists('../data/season_' + str(season)):
            os.makedirs('../data/season_' + str(season))
            os.makedirs('../data/season_' + str(season) + '/teams/')
        fetch_teams_stats(str(season), teams, rewrite, write_game)


def fetch_teams_stats(season, teams, rewrite, write_game):

    features_wanted = {'date_game', 'game_start_time','game_location','opp_name','game_result', 'overtimes', 'pts', 'opp_pts', 'wins', 'losses', 'game_streak'}

    for team in teams:
        try:
            filename = "../data/season_" + season + "/teams/" + team + "_stats.csv"
            url = url_base_team + team + "/" + season + url_season_stats
            if rewrite or not(os.path.exists(filename)):
                df = fetch_team_stats(season, features_wanted, url, write_game)
                df.to_csv(filename, index=False, sep=',', encoding='utf-8')
        except HTTPError:
            print("This {} team was not registered for the season {}\n".format(team, season))


def fetch_team_stats(season, features_wanted, url, write_game):
    team = element_in_string(get_teams(), url)
    print("Building {}_stats.csv... for the season: {}".format(team, season))
    df = pd.DataFrame(columns=features_wanted)
    page = urlopen(url).read()
    soup = BeautifulSoup(page, "html.parser")
    table = soup.find('tbody')
    rows = table.find_all('tr')
    for row in rows:
        if (row.find('th', {"scope":"row"}) != None):
            dico = dict()
            for f in features_wanted:
                cell = row.find("td",{"data-stat": f})
                if f == 'overtimes' and not(cell.text):
                    continue
                a = cell.text.strip().encode()
                text=a.decode("utf-8")
                if f in dico:
                    dico[f].append(text)
                else:
                    dico[f]=[text]
            
            # GAME
            url_team = team
            if dico['game_location'][0] == '@':
                url_team = get_team_from_full_name(dico['opp_name'][0])
            url_date = datetime.strptime(dico['date_game'][0], "%a, %b %d, %Y").strftime("%Y%m%d0")
            url_game = url_base + "boxscores/" + url_date + url_team + ".html"
            game_filename = url_date + url_team + "_stats.csv"
            game_features_wanted = ['mp', 'fg', 'fga', 'fg_pct', 'fg3', 'fg3a', 'fg3_pct', 'ft', 'fta', 'ft_pct', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts', 'plus_minus']
            game_path = 'data/season_' + str(season) + '/teams/' + team
            if write_game and not os.path.exists(game_path):
                os.makedirs(game_path)
            df_game = fetch_game_stats(game_features_wanted, url_game, team)
            if write_game and not(os.path.exists(game_path + "/" + game_filename)):
                df_game.to_csv(game_path + "/" + game_filename, index=False, sep=',', encoding='utf-8')
            df_game = df_game.dropna()
            dico['ast'] = [sum(df_game['ast'].astype('float32'))]
            dico['fg'] = [sum(df_game['fg'].astype('float32'))]
            dico['fga'] = [sum(df_game['fga'].astype('float32'))]
            df_game['fg3'] = df_game['fg3'].apply(lambda x: 0 if x == '' else x)
            dico['fg3'] = [sum(df_game['fg3'].astype('float32'))]
            df_game['fg3a'] = df_game['fg3a'].apply(lambda x: 0 if x == '' else x)
            dico['fg3a'] = [sum(df_game['fg3a'].astype('float32'))]
            dico['ft'] = [sum(df_game['ft'].astype('float32'))]
            dico['fta'] = [sum(df_game['fta'].astype('float32'))]
            dico['orb'] = [sum(df_game['orb'].astype('float32'))]
            dico['drb'] = [sum(df_game['drb'].astype('float32'))]
            dico['trb'] = [sum(df_game['trb'].astype('float32'))]
            dico['stl'] = [sum(df_game['stl'].astype('float32'))]
            dico['blk'] = [sum(df_game['blk'].astype('float32'))]
            dico['tov'] = [sum(df_game['tov'].astype('float32'))]
            dico['pf'] = [sum(df_game['pf'].astype('float32'))]
            df_game['plus_minus'] = df_game['plus_minus'].apply(lambda x: 0 if x == '' else x)
            dico['plus_minus'] = [sum(df_game['plus_minus'].astype('float32'))]
            df = df.append(pd.DataFrame.from_dict(dico), sort=True)
    return df


def fetch_game_stats(features_wanted, url, team):
    df = pd.DataFrame(columns=features_wanted)
    page = urlopen(url).read()
    soup = BeautifulSoup(page, "html.parser")
    count_games = 0
    div = soup.find("div", {"id": "all_box_{}_basic".format(team.lower())})
    table = div.find('tbody')
    rows = table.find_all('tr', {'class': None})
    for row in rows:
        if row.find('th', {"scope":"row"}) != None:
            if row.find("td",{"data-stat": 'reason'}) != None or row.find("td",{"data-stat": 'reason'}) != None:
                df = df.append(pd.Series(), ignore_index=True)
                continue
            dico = dict()
            for f in features_wanted:
                cell = row.find("td",{"data-stat": f})
                a = cell.text.strip().encode()
                text=a.decode("utf-8")
                if f in dico:
                    dico[f].append(text)
                else:
                    dico[f]=[text]
            df = df.append(pd.DataFrame.from_dict(dico), sort=True, ignore_index=True)
    return df

fetch_all_stats([2016, 2020], get_teams())