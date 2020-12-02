
import seaborn as sns
import matplotlib.pyplot as plt  
import pandas as pd

# this code helped analysing the file
# It is not cleaned code
def draw_memory_depth_analysis (df_features, feature_glossary, depth):
    def compute_model_accuracy_by_memory_depth(depth):
        model_accuracy_by_season = pd.DataFrame(range(2002, 2018), columns=['season'])
        return model_accuracy_by_season.apply(lambda x: compute_model_perf(df_features, x, depth ), axis=1)
     
    # Compute model accuracy by depth 
    model_accuracy_by_memory_depth = pd.DataFrame(range(1,19), columns = ['depth']) 
    model_accuracy_by_memory_depth = model_accuracy_by_memory_depth.apply(lambda x: compute_model_accuracy_by_memory_depht(x.depth), axis=1)

    # Rearange the result in one clean dataframe
    df_depth_analysis =  model_accuracy_by_memory_depth.iloc[0][['season', 'accuracy']]
    df_depth_analysis['depth'] = 1
    for i in range(2,18):
        test = model_accuracy_by_memory_depth.iloc[i][['season', 'accuracy']]
        test['depth'] = i
        df_depth_analysis = df_depth_analysis.append(test)
    
    # Draw result 
    sns.set_theme(style="darkgrid")  
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10) 
    sns.lineplot(x="season", y="accuracy", hue='depth', data=df_depth_analysis )
    plt.show() 
     
def show_simulation_result(simulation_data):
    fig, ax = plt.subplots()
    h = simulation_data.team_id.shape[0] / 2
    fig.set_size_inches(10, h)
    g = sns.barplot(x="wins", y="team_name", data=simulation_data[['team_name', 'wins']].sort_values(['wins'], ascending=False), ax=ax)
    plt.xlabel("Number of simulations where the team won the NBA")  
    plt.show()   

def show_cote_comparator(df_a, df_b): 
    df_a.columns = [str(x) + '_bef' for x in df_a.columns]
    df_b.columns = [str(x) + '_aft' for x in df_b.columns]
    df_comparator = pd.merge(df_a, df_b, how='inner', left_on='team_id_bef', right_on='team_id_aft')
    df_comparator = df_comparator[['team_id_bef', 'team_name_bef', 'cote_bef', 'cote_aft']]
    df_comparator = df_comparator.assign(evol_cote= lambda x: x.cote_bef - x.cote_aft) 
    fig, ax = plt.subplots() 
    fig.set_size_inches(10, 8)
    g = sns.barplot(x="evol_cote", y="team_name_bef", data=df_comparator[['team_name_bef', 'evol_cote']].sort_values(['evol_cote'], ascending=False), ax=ax)
    plt.xlabel("Cotes evolution between the two simulation")  
    plt.show()        