from scripts.utils import feature_pipe
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from pathlib import Path
from pathlib import PurePath
import csv


import pickle
default_param = {
           'max_depth': range (2, 10, 1),
            'n_estimators': range(100, 1000, 100),
            'learning_rate': [0.1, 0.01, 0.05]
        }

def xg_boost_pipe(df, param = default_param, grid: bool = False, eval = "roc_auc", name=None ):
    '''
    Training pipe for xgboost
    :param df: Input data set, should go through feature_pipe function first
    :param param: parameter for grid search
    :param grid: bool for grid search activation
    :param eval: set evaluation metric
    :param name: name of model for csv file
    :return: best model and feature importance
    '''

    X = df.loc[:, df.columns != 'Cavitation'].astype('float')
    y = df["Cavitation"]
    if grid == True:
        model = xgb.XGBClassifier(objective = 'binary:logistic')
        print('\n ----------------------------------------------')
        print('\n Initiating Gridsearch, fine tuning parameters!')
        print(f'\n Scoring: {eval}')
        folds = 5


        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)


        grid = GridSearchCV(model, param, n_jobs=5,
                        cv = skf.split(X, y), scoring=eval, verbose=True, refit=True)

        grid.fit(X, y)

        print('\n Done!')
        print('\n Best estimator:')
        print(grid.best_estimator_)
        print('\n  best_score_:')
        print(grid.best_score_)
        print('\n Best parameters:')
        print(grid.best_params_)

        importance = pd.DataFrame({'Variable':X.columns,
                  'Importance':grid.best_estimator_.feature_importances_}).sort_values('Importance', ascending=False)
        best_params = [name,grid.best_params_]

        with open(Path.cwd() / "data/params.csv", 'w') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write a row to the csv file
            writer.writerow(best_params)

        print('\n Feature importance:')
        print(importance)
        # return model and feature importance
        return grid.best_estimator_, importance
    else:
        print('\n -------------------------------')
        print('\n Fitting XGBoost!')
        print('\n No Gridsearch!')
        print(f'\n  {len(X)} samples!')
        model = xgb.XGBClassifier()
        model.fit(X, y)
        importance = pd.DataFrame({'Variable':X.columns,
                  'Importance':model.feature_importances_}).sort_values('Importance', ascending=False)
        print('\n Fitting XGboost finished!')
        return model, importance


def Classification_pipe_all (window_size,folder, evaluation = "accuracy", parameter = default_param , gridsearch: bool = True, save_folder = "models"):
    # list of window sizes (if window_size_eval is set to true, otherwise only 1 window size) ,
    # list of cavitation files, list of no-cavitation files, train test split, evaluation method, window size evaluation
    '''
    Training pipe for xgboost
    :param window_size: Window size for training
    :param evaluation: evaluation metric
    :param parameter: parameter for grid search
    :param gridsearch: bool for grid search activation
    :param save_folder: save folder for model
    :return: best model and feature importance
    '''

    print("Initializing training!")
    if  not isinstance(parameter, dict):
        raise TypeError("Parameter need to be of type dictionary")



    if isinstance(window_size,np.ndarray):
        raise TypeError("This is of type np.ndarray. Only input one variable when split evaluation is set to false.")

    if not isinstance(window_size,(float,int)):
        raise TypeError("Only input one variable when split evaluation is set to false.")

    else:
            # non_cav_len, no_cav_df = feature_pipe(no_cavitation_data, window_size, "No Cavitation")
            # cav_len, cav_df = feature_pipe(cavitation_data, window_size, "Cavitation")
            # final_df = pd.concat([no_cav_df, cav_df])
            # length = cav_len + non_cav_len
        print(f"Window size of {window_size} seconds!")
        length, final_df = feature_pipe(folder, window_size)
        print(f'\n Done! We have {length} samples!')

        model, importance = xg_boost_pipe(final_df, param = parameter, grid = gridsearch, eval = evaluation,name = folder)
        print('\n Feature importance:')
        print(importance)
        if gridsearch:

            filename = Path().resolve()/f"{save_folder}/model_{PurePath(folder).parts[1]}_grid_{window_size}.sav"
            pickle.dump(model, open(filename, 'wb'))
        else:

            filename = Path().resolve()/f"{save_folder}/model_{PurePath(folder).parts[1]}_{window_size}.sav"
            print(filename)
            pickle.dump(model, open(filename, 'wb'))

        return model, importance


if __name__ == "__main__":
   ###best until now #
    #Classification_pipe_all (1, "data/oil_pump_small" , evaluation ="recall", parameter = default_param , window_size_eval = False, gridsearch = True)


    ### pumps
    #Classification_pipe_all(1, "data/oil_vacuum", evaluation="recall", parameter=default_param,window_size_eval=False, gridsearch=True)
    # only oil pump
    #Classification_pipe_all(1, "data/train_oil_vacuum", evaluation="recall", parameter=default_param,  window_size_eval = False, gridsearch = False, save_folder = "data/models/standard/")

    ##testing things debvug
    #Classification_pipe_all(1,"data/electrochem_train_test/train", evaluation ="accuracy", parameter = default_param , window_size_eval = False, gridsearch = True,
                            #save_folder = "data/models/standard/")
    #Classification_pipe_all(1,"data/train_oil_vacuum_add_old_pump/", evaluation ="accuracy", parameter = default_param , window_size_eval = False, gridsearch = True, save_folder = "data/train_oil_vacuum_add_old_pump")
    Classification_pipe_all(1, "data/train_oil_vacuum", evaluation="recall", parameter=default_param, gridsearch=True, save_folder = "data/testing_models")