import math
import pandas as pd
import numpy as np
import datetime as dt
import nbadata


class feature_computer():
    # Game_score is a complexe KPI designed by Dean Olivier reflecting the effect the player had on the match

    resources = None

    def __init__(self):
        self.resources = nbadata.NbaDataSg()

    def compute_game_score(df):
        def game_score_formula(ax):
            _2p = ax.fg - ax._3p
            _pts = _2p * 2 + ax._3p * 3
            gmscore = _pts + 0.4 * ax.fg - 0.7 * ax.fga - 0.4 * (ax.fta - ax.ft) + 0.7 * ax.orb + 0.3 * ax.drb + ax.stl + 0.7 * ax.ast + 0.7 * ax.blk - 0.4 * ax.pf - ax.tov
            return gmscore

        return df.fillna(0).apply(game_score_formula, axis=1)

        # Obsolete even if good results : all of this is random magic numbers

    def compute_histo_gamescore_v1(player_dictionary, player_id, game_date):
        player_histo = player_dictionary.loc[player_id]
        player_histo_filtered = player_histo[player_histo.index < game_date].head(400)
        custom_avg = 0
        if player_histo_filtered.shape[0] > 200:
            avg_new = player_histo_filtered.head(20).game_score.mean()
            avg_middle = player_histo_filtered.head(100).tail(80).game_score.mean()
            avg_old = player_histo_filtered.head(400).tail(300).game_score.mean()
            custom_avg = avg_new * 0.2 + avg_middle * 0.4 + avg_old * 0.4
        elif player_histo_filtered.shape[0] > 10:
            n = player_histo_filtered.shape[0]
            n_8 = int(n * 0.8)
            avg_new = player_histo_filtered.head(n - n_8).game_score.mean()
            avg_old = player_histo_filtered.head(n).tail(n_8).game_score.mean()
            custom_avg = avg_new * 0.3 + avg_old * 0.3
        elif player_histo_filtered.shape[0] != 0:
            custom_avg = player_histo_filtered.game_score.mean() * 0.4
        else:
            custom_avg = 0
        return custom_avg

    def weighted_mean(self, serie):
        max_depth = 200
        s = min(len(serie), max_depth)
        if s == 0:
            return np.nan
        weights = pd.Series(self.resources.histo_weights[:s])
        serie = pd.Series(serie).head(max_depth)
        return np.sum(serie * weights) / np.sum(weights)

    def compute_player_histo_kpi(self, player_dictionary, ax, kpis, last_known_date=dt.date.today()):
        player_id, game_date = ax.player_id, min(ax.datetime, last_known_date)
        player_histo = player_dictionary.loc[player_id]
        player_histo_filtered = player_histo[player_histo.index < game_date].head(100)
        cols = ['histo_{}'.format(x) for x in kpis]
        histo_kpis = pd.Series([self.weighted_mean(player_histo_filtered[kpi].values) for kpi in kpis], index=cols)
        return pd.concat([ax, histo_kpis])

    def compute_players_features(self, df_player_conso, limit_date):
        df_player_histo = self.resources.try_read_pickle('df_player_histo')
        if df_player_histo is not None:
            return df_player_histo
        historized_player_kpis = ['plus_minus', 'ts_pct', 'efg_pct', '_3par', 'ftr', 'orb_pct', 'drb_pct', 'trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct', 'ortg', 'drtg', 'bpm',
                                  'game_score']
        df_player_histo = df_player_conso.apply(lambda x: self.compute_player_histo_kpi(self.resources.player_dictionary, x, historized_player_kpis, limit_date), axis=1)
        df_player_histo = df_player_histo.fillna(df_player_histo.mean())
        df_player_histo.to_pickle('df_player_histo')
        print('players kpi weighted mean computed and saved as df_player_histo')
        return df_player_histo

    def compute_teams_features(self, df, max_date):
        df_games_histo = self.resources.try_read_pickle('df_games_histo')
        if df_games_histo is not None:
            return df_games_histo
        input_kpis = ['tm_efg', 'tm_ftfga', 'tm_ftscore', 'tm_orb', 'tm_ortg', 'tm_pace', 'tm_tov', 'win']
        home_kpis_name = ['histo_home_{}'.format(x) for x in ['efg', 'ftfga', 'ftscore', 'orb', 'ortg', 'pace', 'tov', 'win']]
        away_kpis_name = ['histo_away_{}'.format(x) for x in ['efg', 'ftfga', 'ftscore', 'orb', 'ortg', 'pace', 'tov', 'win']]
        kpis = [input_kpis, home_kpis_name, away_kpis_name]
        df_games_histo = df.apply(lambda x: self.compute_features_from_histo_team(x, kpis, max_date), axis=1)
        df_games_histo.to_pickle('df_games_histo')
        print('teams kpi weighted mean & matchup kpis computed and saved as df_games_histo')
        return df_games_histo

    def compute_features_from_histo_team(self, ax, kpis, last_known_date):
        # Keys
        team_id_away, team_id_home, limit_date = ax.away_id, ax.home_id, min(ax.datetime, last_known_date)

        # Get previous games for each teams
        away_histo = self.resources.get_team_history(team_id_away, limit_date)
        home_histo = self.resources.get_team_history(team_id_home, limit_date)

        # Compute the weighted mean for each of these kpis
        histo_kpis_home = pd.Series([self.weighted_mean(home_histo[kpi].values) for kpi in kpis[0]], index=kpis[1])
        histo_kpis_away = pd.Series([self.weighted_mean(home_histo[kpi].values) for kpi in kpis[0]], index=kpis[2])

        # Last game
        win_pct_same_match = self.resources.get_last_encounters(team_id_home, team_id_away, limit_date).win.mean()
        win_pct_return_match = 1 - self.resources.get_last_encounters(team_id_away, team_id_home, limit_date).win.mean()
        stats_last_encounters = pd.Series([win_pct_same_match, win_pct_return_match], index=['win_pct_same_matchup', 'win_pct_mirror_matchup'])

        histo_kpis = pd.concat([ax, histo_kpis_home, histo_kpis_away, stats_last_encounters])
        return histo_kpis

    def compute_features_from_players(self, df_player_histo, keys):
        historized_player_kpis = ['plus_minus', 'ts_pct', 'efg_pct', '_3par', 'ftr', 'orb_pct', 'drb_pct', 'trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'tov_pct', 'usg_pct', 'ortg', 'drtg', 'bpm',
                                  'game_score']
        cols = ['histo_{}'.format(x) for x in historized_player_kpis]
        data = df_player_histo[keys + cols]
        data_g = data.groupby(keys).agg(np.mean)
        return data_g

    def get_kpi(self, data_game_team, game, team_id, kpi):
        try:
            gs = data_game_team.loc[game, team_id][kpi]
            return gs
        except:
            print('game: {} histo for team_id: {} is missing'.format(game, team_id))
            return np.nan

            # Here, player's game kpis history is put in df_games_histo

    def compute_df_features(self, df_player_histo, df_games_histo, use_pickle=True):
        if use_pickle:
            df_features = self.resources.try_read_pickle('df_features')
            if df_features is not None:
                # Build the glossary
                keys = ['game_id', 'season', 'away_id', 'away_name', 'home_id', 'home_name']
                target = ['ylabel']
                home_features = [x for x in df_features.columns if ('home' in x and not x in keys)]
                away_features = [x for x in df_features.columns if ('away' in x and not x in keys)]
                diff_features = [x for x in df_features.columns if ('diff' in x and not x in keys)]
                feature_glossary = {'df_keys': keys, 'home': home_features, 'diff': diff_features, 'away': away_features, 'target': target}
                return df_features, feature_glossary

        df_features = df_games_histo.copy()

        # Remove useless game outcome kpis
        games_outcome_kpis = ['away_ftscore', 'home_ftscore', 'away_wlratio', 'home_wlratio',
                              'location', 'attendance', 'away_pace', 'away_efg',
                              'away_tov', 'away_orb', 'away_ftfga', 'away_ortg', 'home_pace',
                              'home_efg', 'home_tov', 'home_orb', 'home_ftfga', 'home_ortg']
        df_features.drop(games_outcome_kpis, axis=1, inplace=True)

        # For each game, we want to know what was the mean of the players stats
        df_players_features_history = self.compute_features_from_players(df_player_histo, ['game_id', 'team_id'])

        for col in df_players_features_history.columns:
            df_features[col + '_p_away'] = df_games_histo.apply(lambda x: self.get_kpi(df_players_features_history, x.game_id, x.away_id, col), axis=1)
            df_features[col + '_p_home'] = df_games_histo.apply(lambda x: self.get_kpi(df_players_features_history, x.game_id, x.home_id, col), axis=1)

        # Distinct home & away features
        keys = ['game_id', 'season', 'away_id', 'away_name', 'home_id', 'home_name']
        target = ['ylabel']
        home_features = [x for x in df_features.columns if ('home' in x and not x in keys)]
        away_features = [x for x in df_features.columns if ('away' in x and not x in keys)]

        # Compute diff features as home_feature - away_feature
        for home_feature in home_features:
            away_feature = home_feature.replace('home', 'away')
            diff_feature = home_feature.replace('home', 'diff')
            df_features[diff_feature] = df_features.apply(lambda x: x[home_feature] - x[away_feature], axis=1)

        # Define an helping feature glossary
        diff_features = [x for x in df_features.columns if ('diff' in x and not x in keys)]
        feature_glossary = {'df_keys': keys, 'home': home_features, 'diff': diff_features, 'away': away_features, 'target': target}

        # Fill nan values
        df_features = df_features.fillna(method='bfill')

        # Save the result  
        if use_pickle:
            df_features.to_pickle('df_features')

        # Print('Features computed, now have fun !')
        return df_features, feature_glossary
