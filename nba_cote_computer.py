 
import numpy as np 
import pandas as pd
import nbadata 
import datetime as dt
import feature_computer as ft
import ml_tools as ml
import simulator as sml 
import aside_analysis as aa
 
import warnings
warnings.filterwarnings('ignore')

class CoteComputer(): 
  
    def execute(self):
        # !!---------------------------------------------------------------------------------------------------!!
        # !! From NbaDataSg first call to df_features computing, ~4h30 are necessary to computes all resources !!
        # !! Feel free to use pre-processed pickles in order to skip this long process                         !! 
        # !!---------------------------------------------------------------------------------------------------!!

        # Parameter : 
        # Simulation number for part 1 & 2.
        # ~20 minutes with a laptop are enought to compute 50 000 simulations
        sim_count = 50000

        #------------------------------------------------------------------------------------------------------------------
        #----------------------------- Part 0 : preprocessing & Features extraction    
        #------------------------------------------------------------------------------------------------------------------
        # How possibly this thing could take so long ??
        # For each player in every game we have to identify what was the previous game where he was involved and I
        # do it two times  
        #   - Compute the player team ! We do not have this information at start. We know playerX played this day 
        #     in this game but on which side ? we have to take a look on the ~10 closest game to have a clear idea
        #     (ID) of the player side in a single game.
        #   - Compute the player past stats : Player X played in this team this day, then in an other team... what
        #     are his records ? For each game we want to know the team complete records.
        #
        # Most of this is Hidden in libs. It was very time consuming.  

        # I know singleton are ugly but... all this python data manipulation is ugly too.
        # All of this should have been done with an SGBD.
        # This part generates a lot of "files not found" messages if your are computing the dataset yourself.
        # If you compute them please understand there is 3 ways to get the 2 basics dataset
        #   1. Load them from pickles files named 'df_players' and 'df_games'
        #   2. Download them from S3 repository with a well configurated Boto3 library (the provided method)
        #   3. Download them from S3 repository with login. You will have to create a credentials.txt file in the project
        #      root directory and filled it with {aws_access_key_id}/{aws_secret_access_key}
        resources = nbadata.NbaDataSg()

        start_date_season_2018 = dt.date(2018, 10, 16)
        start_date_playoff_2018 = dt.date(2019, 4, 13)
         
        # Computes features by players
        fc = ft.feature_computer() 
        resources.df_player_histo = fc.compute_players_features(resources.df_player_conso, start_date_season_2018)

        # Computes features by team
        resources.df_games_histo = fc.compute_teams_features(resources.df_games, start_date_season_2018)

        # Merge players data into one dataset containing all imagined features 
        df_features, feature_glossary = fc.compute_df_features(resources.df_player_histo, resources.df_games_histo)


        #------------------------------------------------------------------------------------------------------------------
        #----------------------------- Part 1 :  Regular Season begin, let's predict a cote for each team !             ---
        #------------------------------------------------------------------------------------------------------------------
        # How to do ?
        # 1. Make a game outcome predictor wich gives a probability for home team to beat way team
        #   1.1 Use PCA on df_feature to get synthetic component of team A & B statistics on the previous seasons data
        #   1.2 Train a logistic regression on pca components
        #   1.3 Use the 2018 regular season data as a game agenda and use the model to estimate the home victory porobability
        #       for each game
        # 2. Simulate the NBA thousand of times
        #   2.1 Simulate regular season
        #   2.2 Simulate playoff with the regular season simulation as an input and get a NBA winner
        # 3. Compute Cotes
        #   3.1 If a team win 50% of the simulations, his cote should be 2
        #   3.2 Think about the models limits before betting any euro. 
        #       Hight cotes should not be estimated with this model.

        # PCA is fitted on 2001_2017 time interval : 2000 lack of data and 2018 is supposed unknown
        # 2000 data are usefull for building 2001 & 2002 stats but not more
        df_2001_2017 = df_features[(df_features.season > 2000) & (df_features.season < 2018)] 
        histo_pca = ml.get_pca(df_2001_2017, feature_glossary['home'], 5)
         
        # Each team's features are rotated with the PCA and components added to our data frame as new features 
        df_2001_2018 = df_features[df_features.season > 2000]
        df_2001_2018, feature_glossary = ml.add_pca_conponent_as_features(df_2001_2018, feature_glossary, histo_pca, 5)
         
        # A model is trained with previous season :  
        df_2018 = df_2001_2018[df_2001_2018.season<2018]
        logistic_reg = ml.train_model(df_2018, feature_glossary['pca_components'])

        # Predict outcome probability for each regular season games 
        df_reg_season = df_2001_2018[(df_2001_2018.season==2018) & (df_2001_2018.datetime < start_date_playoff_2018)]
        df_reg_season = df_reg_season.assign(prediction=[x[1] for x in logistic_reg.predict_proba(df_reg_season[feature_glossary['pca_components']])])

        # Compute all the 1 to 1 matchups 
        df_matchups = df_2001_2018[df_2001_2018.season==2018]
        df_matchups = df_matchups.assign(prediction=[x[1] for x in logistic_reg.predict_proba(df_matchups[feature_glossary['pca_components']])])
        df_matchups = df_matchups[['home_id', 'away_id', 'prediction']].groupby(['home_id', 'away_id']).apply(np.mean)

        # Here a NBA simulator. Many details and rules are missing but it is still very realistic.
        sim = sml.Simulator() 
         
        # Lauch X simulations
        df_win_sim = pd.DataFrame(range(sim_count), columns=['wins'])
        df_win_sim['team_id']= df_win_sim.apply(lambda x: sim.simulate_nba(df_reg_season, df_matchups), axis=1)
          
        # Count wins by teams 
        df_win_sim = df_win_sim.groupby(['team_id']).count().sort_values(['wins'], ascending=False).reset_index()

        # Make it readable with team names
        df_cotes_before = pd.merge(sim.team_div_conf, df_win_sim, how='inner', left_on=['team_id'], right_on=['team_id'])

        # Compute Cote 
        df_cotes_before = df_cotes_before.assign(win_pct=lambda x: x.wins / sim_count) 
        df_cotes_before = df_cotes_before.assign(cote=lambda x: round(1 / x.win_pct,2) ) 

        print('A cote has been evaluated for each team before the regular season')

        # Export Result
        df_cotes_before.to_csv('cote_nba_before_regular_season.csv', decimal=',', sep=';', index=False)
        aa.show_simulation_result(df_cotes_before)  

        #------------------------------------------------------------------------------------------------------------------
        #----------------------------- Part 2 :  Regular Season is over, let's predict an updated cote for each team !  ---
        #------------------------------------------------------------------------------------------------------------------
        # Here, we use the same methodology with the followings changes :
        #  1. Regular season games are used as training data set.
        #     - We won't recompute all the data here, I did it for you using the very same process with nbadata just by 
        #       changing the 'max known date' in 'fc.compute_players_features' in & 'fc.compute_teams_features'. Of course I could have recompute only the regular season data but I did not want to 
        #       spend more time on this painfull data processing part.
        #     - I could have drop all the data post regular season, it would have been more elegant. Id did not in order to 
        #       facilitate the matchup definition.
        #       
        #  2. The simulator will simulate only the playoff simulator. 
        #     - Two files are needed ranking_est.csv & ranking_ouest.csv and give the tournament structure.
        #     - Fill free to play with the the simulations number  
        #
            
        # Get an updated features dataframe where all kpi's take into account the 2018 regular season
        df_features_playoff = pd.read_csv('df_features_playoff.csv', sep=';')
        df_features_playoff['datetime'] = df_features_playoff['datetime'].apply(pd.to_datetime)

        # New features are computed using PCA rotation 
        df_2010_2018 = df_features_playoff[df_features_playoff.season > 2010]
        df_2010_2018, feature_glossary = ml.add_pca_conponent_as_features(df_2010_2018, feature_glossary, histo_pca, 5)
         
        # A model is trained with previous games including 2018 regular season:  
        df_training = df_2010_2018[df_2010_2018['datetime'] < pd.to_datetime(start_date_playoff_2018)]
        logistic_reg = ml.train_model(df_training, feature_glossary['pca_components'])

        # Predict outcome probability for each 2018 game
        df_2018 = df_2010_2018[df_2010_2018.season == 2018]
        df_2018 = df_2018.assign(prediction=[x[1] for x in logistic_reg.predict_proba(df_2018[feature_glossary['pca_components']])])

        # Now we have a win probability for every matchup 
        df_matchups = df_2018[['home_id', 'away_id', 'prediction']].groupby(['home_id', 'away_id']).apply(np.mean)

        # Use the simulator
        sim = sml.Simulator() 

        # Load 2018 regular season ranking by conferences
        ranking_E, ranking_O = pd.read_csv('ranking_est.csv', sep=';'), pd.read_csv('ranking_ouest.csv', sep=';')

        # Lauch X simulations 
        playoff_simulation = pd.DataFrame(range(sim_count), columns=['wins'])
        playoff_simulation['team_id']= playoff_simulation.apply(lambda x: sim.simulate_playoff(df_matchups, ranking_E, ranking_O), axis=1)

        # Count wins by teams 
        playoff_simulation_g = playoff_simulation.groupby(['team_id']).count().sort_values(['wins'], ascending=False).reset_index()

        # Make it readable with team names
        df_cotes_after = pd.merge(sim.team_div_conf, playoff_simulation_g, how='inner', left_on=['team_id'], right_on=['team_id'])

        # Compute Cote 
        df_cotes_after = df_cotes_after.assign(win_pct=lambda x: x.wins / sim_count) 
        df_cotes_after = df_cotes_after.assign(cote=lambda x: round(1 / x.win_pct,2) )  

        print('A cote has been evaluated for each team before the playoff')

        # Export Result 
        df_cotes_after.to_csv('cote_nba_before_playoff.csv', decimal=',', sep=';', index=False) 
        aa.show_simulation_result(df_cotes_after)  

        # How evolve the cotes after play off ?
        aa.show_cote_comparator(df_cotes_before, df_cotes_after)

        