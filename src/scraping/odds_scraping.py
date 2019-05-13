from urllib.request import urlopen, Request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from tools.tools import get_team_from_full_name, get_team_from_short_name
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import os


base_url = 'https://www.rivalo.com/en/sports-results/basketball/usa/gbhhdab-nba/'


def read_odds(season, dates):
    df = None
    dir_name = '../data/season_{}/odds/'.format(str(season))
    if not(os.path.exists(dir_name + 'odds.csv')):
        if not(os.path.exists(dir_name)):
            os.makedirs(dir_name)
        weeks = dates_to_weeks(dates)
        df_to_fill = dates.reset_index(level=['games', 'date_game'])[['games', 'date_game']]
        df = fetch_all_odds(weeks, base_url, ['odd_home_team', 'odd_away_team'], season, dir_name + 'odds.csv', df_to_fill)
    else:
        df = pd.read_csv(dir_name + 'odds.csv')
    return df


def fetch_all_odds(weeks, base_url, features_wanted, season, filename, df_to_fill):

    date1 = None
    date2 = None

    df_to_fill["odd_home_team"] = np.zeros(df_to_fill.shape[0], dtype='float32')
    df_to_fill["odd_away_team"] = np.zeros(df_to_fill.shape[0], dtype='float32')

    for nb_week, week in weeks:

        url = base_url

        if nb_week > 30:
            url += str(season - 1)
        else:
            url += str(season)

        url += '/kw' + str(nb_week)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3'}
        req = Request(url=url, headers=headers)
        page = urlopen(req).read()
        soup = BeautifulSoup(page, "html.parser")

        div = soup.find("div", {"id": "resultsArchive"})
        rows = div.find_all("div", {"class": None})[2].contents
        rows = list(filter(lambda x: x != '\n', rows))

        for row in rows:
            date_parsed = row.find_all("div", {"class":"col_1_archive bold left"})
            if len(date_parsed):
                date_tmp = date_parsed[0].contents[0].strip().split(' ')[1].split('.')
                year = season - 1 if int(date_tmp[1]) > 7 else season
                date1 = str(year) + "-" + date_tmp[1] + "-" + date_tmp[0]
                date1_datetime = datetime.strptime(date1, "%Y-%m-%d") - timedelta(days=1)  
                date2 = date1_datetime.strftime("%Y-%m-%d")
                print(date1, date2)
            else:
                game_name = row.find_all("div", {"class":"col_3_archive left"})[0].contents[0]
                rates = row.find_all("div", {"class":"col_5 left "})
                rates += row.find_all("div", {"class":"col_5 left bold"})
                odd_home_team, odd_away_team = parse_rates(rates)
                print(game_name_to_short(game_name))
                df_to_fill.loc[(df_to_fill['games'] == game_name_to_short(game_name)) & ((df_to_fill['date_game'] == date1) | (df_to_fill['date_game'] == date2)), 'odd_home_team'] = odd_home_team
                df_to_fill.loc[(df_to_fill['games'] == game_name_to_short(game_name)) & ((df_to_fill['date_game'] == date1) | (df_to_fill['date_game'] == date2)), 'odd_away_team'] = odd_away_team
    df_to_fill.to_csv(filename, index=False, sep=',', encoding='utf-8')
    return df_to_fill


def game_name_to_short(s):
    new_team = ""
    tmp = s.split('-')
    home_team = remove_trailing_spaces(tmp[0])
    away_team = remove_trailing_spaces(tmp[1])
    home_team_first = home_team.split(' ')[-1]
    home_team_second = ' '.join(home_team.split(' ')[:-1])
    away_team_first = away_team.split(' ')[-1]
    away_team_second = ' '.join(away_team.split(' ')[:-1])

    if get_team_from_full_name(home_team_first) != None:
        new_team += get_team_from_full_name(home_team_first) + "-"
    else:
        new_team += get_team_from_full_name(home_team_second) + "-"
    
    if get_team_from_full_name(away_team_first) != None:
        new_team += get_team_from_full_name(away_team_first)
    else:
        new_team += get_team_from_full_name(away_team_second)
    return new_team


def remove_trailing_spaces(s):
    arr = [x for x in s.split(' ') if x != '']
    res = ""
    for i in range(len(arr)):
        if i != 0:
            res += " "
        res += arr[i]
    return res


def parse_rates(l):
    rates = []
    reverse = False
    for i in range(len(l)):
        rate = l[i].contents[0].split('\\')[0][:-1]
        if rate != '0,00':
            rates.append(rate)
        elif i == 0:
            reverse = True
    if reverse:
        rates.reverse()
    return rates[0], rates[1]



def dates_to_weeks(dates):
    weeks = []
    week = []
    nb_week_tmp = 0
    for date, new_df in dates.groupby(level=0):
        new_date = datetime.strptime(date, '%Y-%m-%d')
        nb_week = new_date.isocalendar()[1]
        if nb_week_tmp != nb_week:
            if not((nb_week_tmp, week) == (0, [])):
                weeks.append((nb_week_tmp, week))
            week = []
            nb_week_tmp = nb_week
        for game in new_df.index:
            tmp = game[1].split('-')
            new_game = get_team_from_short_name(tmp[0]) + ' - ' + get_team_from_short_name(tmp[1])
            if get_team_from_short_name(tmp[1]) == "Minnesota Timberwolves":
                new_game = get_team_from_short_name(tmp[0]) + ' - ' + "Minnesota Timber."
            elif get_team_from_short_name(tmp[0]) == "Minnesota Timberwolves":
                new_game = "Minnesota Timber." + ' - ' + get_team_from_short_name(tmp[1])
            week.append(new_game)
    weeks.append((nb_week_tmp, week))
    return weeks