import numpy as np  
import datetime as dt
import pandas as pd
import random

class Simulator():
    # Contains the keys team_id, division_id, conf_id
    team_div_conf = None
    
    def __init__(self):  
        # Compute Resources used for SEASON & PLAY OFF Simulations 
        self.team_div_conf = pd.read_csv('team_div_conf.csv', sep=';') 

    # SEASON SIMULATION
    def simulate_season(self, df_season):
        # For each game, roll a dice to define a winner accordingly to probabilities for team A to win vs team B 
        df_season['random'] = df_season.apply(lambda x: random.random(), axis=1)
        df_season = df_season.assign(wins_home=lambda x: (x.random < x.prediction) + 0) 
        df_season = df_season.assign(wins_away=lambda x: (x.random >= x.prediction) + 0) 

        # Compute victory count by team
        home_wins = df_season[['home_id', 'wins_home']].groupby(['home_id']).agg(np.sum).reset_index()
        away_wins = df_season[['away_id', 'wins_away']].groupby(['away_id']).agg(np.sum).reset_index()
        df_wins_by_team = pd.merge(home_wins, away_wins, how='inner', left_on=['home_id'], right_on=['away_id']).assign(wins=lambda x: x.wins_home + x.wins_away)[['home_id', 'wins']].rename(columns={'home_id':'team_id'})

        # Merge team performances during the season to divisions / conferences data
        df_wins_by_team = pd.merge(self.team_div_conf, df_wins_by_team, how='inner', left_on=['team_id'], right_on=['team_id'])

        # Compute & return the conferences ranking
        df_conf_ranking = df_wins_by_team.sort_values(['conf','wins'], ascending=False)
        ranking_E = df_wins_by_team[df_wins_by_team.conf=='E'].sort_values(['wins'], ascending=False).reset_index(drop=True).head(8).reset_index(drop=True)
        ranking_O = df_wins_by_team[df_wins_by_team.conf=='O'].sort_values(['wins'], ascending=False).reset_index(drop=True).head(8).reset_index(drop=True)
        return ranking_E, ranking_O
        
    # Math stuff. It gives the probability to win the best of 7 if you have the probability to win a single match 
    def best_of_7(self, proba_win_home, proba_win_away):
        P1, P2 = proba_win_home,  proba_win_away
        # Binomial random law
        proba_win_bestof7 = pow(P1,4) + 4 * pow(P1,3) * (1 - P1) * (1 - pow(1 - P2, 3)) + 6 * pow(P1,2) * pow(1 - P1, 2) * (3 * pow(P2,2)* (1 - P2) + pow(P2,3)) + 4 * P1 * pow(1-P1,3) * pow(P2,3)
        return proba_win_bestof7

    # Randomly define a winner accordingly to probabilities for team A to win vs team B in a best of 7
    def get_bestof7_winner(self, df_matchups, team_a, team_b):
        proba_A_win_home = df_matchups.loc[team_a, team_b].prediction
        proba_A_win_away = 1 - df_matchups.loc[team_b, team_a].prediction

        proba_A_win = self.best_of_7(proba_A_win_home, proba_A_win_away)
        rd = random.random()
        
        if rd < proba_A_win:  
            return team_a 
        return team_b
     
    # PLAY OFF SIMULATION 
    def simulate_playoff(self, df_matchups, ranking_E, ranking_O): 
        # 1rst tour  
        half_final = pd.DataFrame(index=range(8), columns=['team_id'])
        for i in range(4):
            half_final.iat[i,0] = self.get_bestof7_winner(df_matchups, ranking_E.iat[i,0], ranking_E.iat[7-i,0])
        for i in range(4):
            half_final.iat[4+i,0] = self.get_bestof7_winner(df_matchups, ranking_O.iat[i,0], ranking_O.iat[7-i,0])

        # Half Final Simulation
        conf_final = pd.DataFrame(index=range(4), columns=['team_id'])
        for i in range(4):
            conf_final.iat[i,0] = self.get_bestof7_winner(df_matchups, half_final.iat[2*i,0], half_final.iat[2*i+1,0]) 

        # Conf Final     
        nba_final = pd.DataFrame(index=range(2), columns=['team_id'])
        for i in range(2):
            nba_final.iat[i,0] = self.get_bestof7_winner(df_matchups, conf_final.iat[2*i,0], conf_final.iat[2*i+1,0]) 

        winner = self.get_bestof7_winner(df_matchups, nba_final.iat[0,0], nba_final.iat[1,0])
        return winner
        
    def simulate_nba(self, df_reg_season, df_matchups):
        sim_ranking_E, sim_ranking_O = self.simulate_season(df_reg_season) 
        return self.simulate_playoff(df_matchups, sim_ranking_E, sim_ranking_O)    