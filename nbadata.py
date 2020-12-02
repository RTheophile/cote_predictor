import math
import os.path
import datetime as dt
import pandas as pd
import numpy as np
import statistics as st

from nbaloading import DataLoader  

class NbaData(): 
    # Basic Resources
    df_games = None 
    df_players = None
    
    # Util
    histo_weights = None
    
    # Player related resources
    player_dictionary = None
    df_game_player_team = None
    df_player_conso = None
    df_player_histo = None
    
    # Team related resources
    team_history_dic = None
    df_encounters = None
    df_games_histo = None
     
    def __init__(self):
        self.df_games = self.load_games_data() 
        self.df_players = self.load_players_data() 
        self.compute_resources()
        
    def compute_resources(self): 
        self.compute_weights()  
        self.df_game_player_team = self.try_read_pickle('df_game_player_team')  
        if self.df_game_player_team is None:
            self.compute_game_player_team()
            
        self.df_player_conso = self.try_read_pickle('df_player_conso')  
        if self.df_player_conso is None:    
            self.compute_player_conso()
            
        self.compute_player_dictionary()
        
        self.df_encounters = self.try_read_pickle('df_encounters')  
        if self.df_encounters is None: 
            self.compute_encounters_table()
            
        self.team_history_dic = self.try_read_pickle('team_history_dic')  
        if self.team_history_dic is None: 
            self.compute_all_teams_history()    
        
    def load_games_data(self):
        df_games = DataLoader().load('df_games') 
        return self.get_cleaned_games_data(df_games)
        
    def load_players_data(self):
        df_players = DataLoader().load('df_players')
        return df_players
        
    def try_read_pickle(self, file_name):
        try:
            df = pd.read_pickle(file_name) 
            print('Info : {} read from pickle'.format(file_name))
            return df
        except:
            print('Info : {} not found'.format(file_name))
            return None
        
    def get_cleaned_games_data(self, df_games): 
        # Analysis does not take into account tiers 
        useless_columns = ['away1', 'away2', 'away3', 'away4', 'away1_ot', 'away2_ot', 'away3_ot'
                           , 'away4_ot' , 'home1', 'home2', 'home3', 'home4', 'home1_ot', 'home2_ot'
                           , 'home3_ot' , 'home4_ot', 'official1', 'official2', 'official3']
        df_games.drop(columns=useless_columns, axis=1, inplace = True)
        # datetime processing
        df_games['datetime'] = df_games['datetime'].apply(lambda x: dt.datetime.strptime(x[:10], '%Y-%m-%d').date())

        # sort dataset on datetime
        df_games = df_games.sort_values(['datetime']).reset_index(drop = True) 
        return df_games
        
    # Used for averaging kpi's of past games     
    def compute_weights(self): 
        # Optimal lambda_parameter has been computed with a logistic regression
        lambda_parameter = 0.04
        self.histo_weights = [1 + math.exp(-x*lambda_parameter) for x in range(1000)]
        print('histo_weights computed')
        
    # Used to get quickly any player's data 
    def compute_player_dictionary(self): 
        self.player_dictionary = self.df_player_conso.groupby(['player_id', 'datetime']).agg(np.mean).sort_values(['player_id', 'datetime'], ascending=False)
        print('Player_dictionary computed') 
        
    def compute_player_conso(self):
        print('player dataset is filled')
        # Data consolidation : all the kpi's are merge by players
        
        df_gp = pd.merge(self.df_players, self.df_games, how='inner', left_on=['game_id'], right_on=['game_id'])
        
        df_player_conso = pd.merge(df_gp, self.df_game_player_team, how='inner', left_on=['game_id', 'player_id'], right_on=['game_id', 'player_id'])
        
        # 'is_home' computing 
        df_player_conso['is_home'] = df_player_conso['team_id'] == df_player_conso['home_id'] 
        df_player_conso['is_home'] = df_player_conso['is_home'] + 0 
        
        def compute_game_score(df):
            def  game_score_formula(ax):
                _2p = ax.fg - ax._3p
                _pts = _2p*2 + ax._3p*3   
                gmscore = _pts + 0.4*ax.fg - 0.7*ax.fga - 0.4*(ax.fta-ax.ft)+ 0.7*ax.orb + 0.3*ax.drb + ax.stl + 0.7*ax.ast + 0.7*ax.blk - 0.4*ax.pf - ax.tov
                return gmscore 
            return df.fillna(0).apply(game_score_formula, axis=1) 
        df_player_conso['game_score'] = compute_game_score(df_player_conso)
           
        self.df_player_conso = df_player_conso
        df_player_conso.to_pickle('df_player_conso')
        print('Players consolidated dataset computed')
        
    def compute_game_player_team(self): 
        # Get the last matches where the player played 
        def get_closest_player_matchs(df, match_day, return_count):
            df['time_distance'] = abs(df['datetime'] - match_day)
            return df.sort_values(['time_distance']).head(return_count)

        # Deduce the team's id of the player by taking a look at the last matches he played 
        def get_player_team(df):
            teams_ids = list(df.away_id) + list(df.home_id)
            return st.mode(teams_ids)
            
        used_columns = ['player_id', 'datetime', 'game_id', 'home_id', 'away_id']

        # Games & player's stats merging
        game_player_team = pd.merge(self.df_players, self.df_games, how='inner', left_on=['game_id'], right_on=['game_id'])[used_columns]

        # Initializing team_id column
        game_player_team['team_id'] = -1

        df_players_dic = game_player_team[used_columns].groupby(['player_id', 'game_id']).agg(lambda x: x.iloc[0])
        team_id_idx = list(game_player_team.columns).index('team_id')

        # For each player-game key, player'team is identified
        for i in range(game_player_team.shape[0]):
            player_id = game_player_team.loc[i,'player_id']  
            match_day = game_player_team.loc[i,'datetime']

            # Team's player is identified by taking a look on the nearest games
            df_last_games = get_closest_player_matchs(df_players_dic.loc[player_id], match_day, 10)
            team_id = get_player_team(df_last_games)    
            game_player_team.iat[i,team_id_idx] = team_id
 
        # saving the result. 
        game_player_team = game_player_team[['game_id', 'player_id', 'team_id']]
        game_player_team.to_pickle('df_game_player_team') 
        
        self.df_game_player_team = game_player_team
        
    def compute_encounters_table(self): 
        team_ids = self.df_games.home_id.unique()
        team_ids.sort()
        data = self.df_games[['season', 'datetime', 'ylabel']]
        self.df_encounters = pd.DataFrame(None, index=range(31), columns=range(31))
        data.columns = ['season', 'datetime', 'win']
        for i in team_ids:
            for j in team_ids:
                if i==j:
                    continue
                self.df_encounters.iat[i,j] = data[(self.df_games.home_id == i) & (self.df_games.away_id == j)].reset_index(drop=True).copy()
        self.df_encounters.to_pickle('df_encounters')
        print('encounters_table computed')
        
    def get_last_encounters(self, home_id, away_id, max_date): 
        key = home_id, away_id 
        last_encounters = self.df_encounters.iat[key]
        return last_encounters[last_encounters.datetime < max_date].tail(5)   
     
    def compute_team_history(self, df_games, team_id): 
        # Get game history where they played at home
        df_histo_home = df_games[df_games.home_id == team_id]

        # Rename & sort columns
        cols = [x.replace('home_', 'tm_').replace('away_', 'opp_') for x in df_histo_home.columns]
        df_histo_home.columns = cols
        cols.sort()
        df_histo_home = df_histo_home[cols]
        df_histo_home = df_histo_home.assign(is_home=0)

        # Get game history where they played away
        df_histo_away = df_games[df_games.away_id == team_id]

        # Rename & sort columns
        cols = [x.replace('away_', 'tm_').replace('home_', 'opp_') for x in df_histo_away.columns]
        df_histo_away.columns = cols
        cols.sort()
        df_histo_away = df_histo_away[cols] 
        df_histo_away = df_histo_away.assign(is_home=1)

        df_histo = pd.concat([df_histo_away, df_histo_home]).sort_values(['datetime'], ascending = False).reset_index(drop=True)

        # Compute the winning team 
        df_histo['win'] = df_histo['tm_ftscore'] > df_histo['opp_ftscore']
        df_histo['win'] = df_histo['win'] + 0
        
        return df_histo
     
    def compute_all_teams_history(self):
        team_ids = self.df_games.home_id.unique()
        team_ids.sort()  
        self.team_histories_dic = {team_id: self.compute_team_history(self.df_games, team_id) for team_id in team_ids} 
        print('team_history_dic computed')
     
    def get_team_history(self, team_id, max_date):
        df_team_history = self.team_histories_dic[team_id]
        df_team_history = df_team_history[df_team_history['datetime'] < max_date]
        df_team_history = df_team_history.sort_values(['datetime'], ascending = False).reset_index(drop=False)
        return df_team_history 
        
def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class NbaDataSg(NbaData):
    pass
