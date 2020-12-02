
from sklearn.linear_model import LogisticRegression  
from sklearn.decomposition import PCA  
import numpy as np  
import datetime as dt
import pandas as pd

# Provide a fitted PCA 
def get_pca(data, columns, n_components):     
    data = data.dropna().reset_index(drop=True) 
    pca = PCA(n_components=n_components)
    X = data[columns]
    pca.fit(X)
    return pca

# Compute the features components after pca rotation and add them to the provided dataframe
def add_pca_conponent_as_features(df, feature_glossary, pca, n_components):
    df = df.fillna(method='bfill').fillna(df.mean()).reset_index(drop=True)
    
    # Components columns are computed using the provided pca
    component_columns = ['Component_'+ str(x) for x in range(n_components)]
    Xt_home = pd.DataFrame(pca.transform(df[feature_glossary['home']]), columns=[x + str('_home') for x in component_columns])
    Xt_away = pd.DataFrame(pca.transform(df[feature_glossary['away']]), columns=[x + str('_away') for x in component_columns])

    # Components columns are added to our original dataframe
    for i in range(n_components):
        df[Xt_home.columns[i]] = Xt_home[Xt_home.columns[i]]
        df[Xt_away.columns[i]] = Xt_away[Xt_away.columns[i]]
    
    # feature_glossary is updated. This thing is really helping in a dataframe containing more than 80 columns
    feature_glossary['pca_components'] = list(Xt_home.columns) + list(Xt_away.columns)
    return df, feature_glossary
      
def train_and_predict(train, test, features_name):
    X_train, X_test = train[features_name], test[features_name]
    y_train, y_test = train.ylabel, test.ylabel
    logistic_reg = LogisticRegression(random_state=0, penalty='none', solver = 'lbfgs', max_iter=500) 
    logistic_reg.fit(X_train, y_train)
    y_pred = logistic_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return logistic_reg, accuracy, logistic_reg.coef_[0]

# Just train a logisqtic regression. I used grid search for optimizing.
def train_model(df, features_name):
    X_train, y_train = df[features_name], df.ylabel 
    logistic_reg = LogisticRegression(random_state=0, penalty='none', solver = 'lbfgs', max_iter=500) 
    logistic_reg.fit(X_train, y_train)
    return logistic_reg