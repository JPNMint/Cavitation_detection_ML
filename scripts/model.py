from scripts.utils import feature_pipe
from sklearn.model_selection import GridSearchCV,train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score

def xg_boost_pipe(df,test_split, param, grid: bool = False, eval = "roc_auc"):
    train, test = train_test_split(df, test_size = test_split, random_state = 1623806)
    X_train = train.loc[:, df.columns != 'Cavitation'].astype('float')
    X_test = test.loc[:, df.columns != 'Cavitation'].astype('float')
    y_train = train["Cavitation"]
    y_test = test["Cavitation"]
    print(f'\n Train set: {len(train)} samples!')
    print(f'\n Test set: {len(test)} samples!')
    if grid == True:
        model = xgb.XGBClassifier(objective = 'binary:logistic')
        print('\n ----------------------------------------------')
        print('\n Initiating Gridsearch, fine tuning parameters!')
        print(f'\n Scoring: {eval}')
        grid = GridSearchCV(model, param, n_jobs=5,
                        cv = 10,
                           #StratifiedKFold( n_splits=5, shuffle=True),
                           scoring=eval, verbose=True, refit=True)

        grid.fit(X_train, y_train)
        #Output
        print('\n Done!')
        print('\n Best estimator:')
        print(grid.best_estimator_)
        print('\n  best_score_:')
        print(grid.best_score_)
        print('\n Best parameters:')
        print(grid.best_params_)

        predicted_y = grid.predict(X_test)

        importance = pd.DataFrame({'Variable':X_train.columns,
                  'Importance':grid.best_estimator_.feature_importances_}).sort_values('Importance', ascending=False)
        print('\n Feature importance:')
        print(importance)

        return train, test, grid.best_estimator_, predicted_y, importance
    else:
        print('\n -------------------------------')
        print('\n Fitting XGBoost!')
        print(f'\n Train set: {len(train)} samples!,  Test set: {len(test)} samples!')
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        predicted_y = model.predict(X_test)
        evaluation = metrics.classification_report(y_test, predicted_y, output_dict=True)

        return train, test, model , predicted_y, evaluation


default_param = {
           'max_depth': range (2, 10, 1),
            'n_estimators': range(100, 1000, 100),
            'learning_rate': [0.1, 0.01, 0.05]
        }


def Classification_pipe_all (window_size, cavitation_data, no_cavitation_data, test_split_ratio, evaluation ="accuracy", parameter = default_param , window_size_eval: bool = False, gridsearch: bool = True):
    # list of window sizes (if window_size_eval is set to true, otherwise only 1 window size) ,
    # list of cavitation files, list of no-cavitation files, train test split, evaluation method, window size evaluation
    #read in files

    if  not isinstance(parameter, dict):
        raise TypeError("Parameter need to be of type dictionary")
    #parameter = {
    #        'max_depth': range (2, 10, 1),
    #        'n_estimators': range(100, 1000, 100),
    #        'learning_rate': [0.1, 0.01, 0.05]
    #    }
    if window_size_eval == True:

        print(f"Window size evaluation is set to true, range is {window_size}")
        print(f"\n Evaluation method is {evaluation}")
        f1_result = []
        acc_result = []
        sample_size_list = []
        splits_list = []

        # loop all splits and get scores
        if isinstance(window_size,(float,int,list)):
            raise TypeError("Invalid window sizes!")
        for splits in window_size:
            splits_list.append(splits)
            non_cav_len, no_cav_df = feature_pipe(no_cavitation_data,splits,"No Cavitation")
            cav_len, cav_df = feature_pipe(cavitation_data,splits,"Cavitation")
            final_df = pd.concat([no_cav_df, cav_df])
            length = cav_len + non_cav_len
            sample_size_list.append(length)
            print(f"{splits} second splits")

            #start xg boost no grid
            train, test, model, predicted_y, eval = xg_boost_pipe(final_df, test_split_ratio, parameter, grid = False)

            print(f'\n Classification report on unseen test set:')
            print(f"\n accuracy:{eval['accuracy']}, f1: {eval['macro avg']['f1-score']}")
            f1_result.append(eval['macro avg']['f1-score'])
            acc_result.append(eval["accuracy"])
            print(f'\n -------------------------------------')
        zipped = list(zip(splits_list, sample_size_list, f1_result, acc_result,  ))
        output = pd.DataFrame(zipped, columns=['Splits_in_sec', 'Sample_size', 'f1', 'accuracy'])


        #get best model using evaluator

        optim = output.loc[output[evaluation].idxmax(), 'Splits_in_sec']

        print(f"\n RESULT: window size evaulation based on {evaluation}, best split size is {optim} seconds!")

        non_cav_len, no_cav_df = feature_pipe(no_cavitation_data, optim, "No Cavitation")
        cav_len, cav_df = feature_pipe(cavitation_data, splits, "Cavitation")
        final_df = pd.concat([no_cav_df, cav_df])

        # start xg boost gridsearch for final model
        train, test, model, predicted_y, importance = xg_boost_pipe(final_df, test_split_ratio, parameter, grid=True, eval = evaluation)

        print(f'\n Classification report on unseen test set:')
        print(f"\n accuracy:{eval['accuracy']}, f1: {eval['macro avg']['f1-score']}")

        return train, test, model, output

    # if no split evaluation, only grid search
    else:
        if isinstance(window_size,np.ndarray):
            raise TypeError("This is of type np.ndarray. Only input one variable when split evaluation is set to false.")

        if not isinstance(window_size,(float,int)):
            raise TypeError("Only input one variable when split evaluation is set to false.")
        else:
            non_cav_len, no_cav_df = feature_pipe(no_cavitation_data, window_size, "No Cavitation")
            cav_len, cav_df = feature_pipe(cavitation_data, window_size, "Cavitation")
            final_df = pd.concat([no_cav_df, cav_df])
            length = cav_len + non_cav_len
            print(f'\n Done! We have {length} samples!')

            train, test, model, predicted_y, importance = xg_boost_pipe(final_df, test_split_ratio, parameter, grid = gridsearch, eval = evaluation)



            return train, test, model, importance
