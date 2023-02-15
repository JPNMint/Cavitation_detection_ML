import pickle
from pathlib import Path
from scripts.utils import feature_pipe
from sklearn import metrics
import numpy as np
import pandas as pd
from scripts.xg_model import Classification_pipe_all

def evaluation(model_name, testset_folder, window_size):
    '''
    Evaluate performance of model
    :param model_name: model to laod
    :param testset_folder: test set folder
    :param window_size: window size of trained model

    '''
    path = Path.cwd()/"data"/model_name
    #path_split = PurePath(path).parts
    model = pickle.load(open(path, 'rb'))
    length, final_df = feature_pipe(testset_folder, window_size)
    print(f'\n Done! We have {length} test samples!')
    X_test = final_df.loc[:, final_df.columns != 'Cavitation'].astype('float')
    y_test = final_df["Cavitation"]
    predicted_y = model.predict(X_test)
    print(y_test)
    print(predicted_y)
    evaluation = metrics.classification_report(y_test, predicted_y, output_dict=True)
    print(evaluation)

def get_feature_importance(df, model_name):
    '''
    Evaluate performance of model
    :param df: dataframe for column name
    :param model_name: model to laod
    :return: feature importance

    '''

    path = Path.cwd()/"data"/model_name
    model = pickle.load(open(path, 'rb'))
    X = df.loc[:, df.columns != 'Cavitation'].astype('float')
    importance = pd.DataFrame({'Variable': X.columns,
                               'Importance': model.best_estimator_.feature_importances_}).sort_values('Importance',ascending=False)
    print(importance)
    return importance

def evaluation_save(model_location, testset_folder, window_size):
    '''
    Evaluate performance of model and returns as dict
    :param model_location: model to laod
    :param testset_folder: test set folder
    :param window_size: window size of trained model
    :return: test set, model, window size, acc, prec, rec, f1

    '''

    print(f"Evaluating model {model_location}!")
    path = model_location#Path.cwd()/"data"/model_name
    model = pickle.load(open(path, 'rb'))
    length, final_df = feature_pipe(testset_folder, window_size)
    print(f'\n Done! We have {length} test samples!')
    X_test = final_df.loc[:, final_df.columns != 'Cavitation'].astype('float')
    y_test = final_df["Cavitation"]
    predicted_y = model.predict(X_test)
    evaluation = metrics.classification_report(y_test, predicted_y, output_dict=True)
    #print(evaluation["1"])

    return {"test": testset_folder, "model": model_location,"window_size_sec":window_size,"accuracy": evaluation["accuracy"], "precision":evaluation["1"]["precision"],"recall":evaluation["1"]["recall"], "f1":evaluation["1"]["f1-score"]}


def eval_all_script(data,data2,data3, traindatalist, traindatalocation ="data", save_folder_model =  "data/models ", save_folder_result =  "data/results" ,data4=None ):
    '''
    Train models and evaluate performance of multiple models on multiple test sets
    Multiple window sizes tested
    :param data1: test data set 1
    :param data2: test data set 2
    :param data3: test data set 3
    :param traindatalist: train set list to train
    :param save_folder_model: where to save all trained models
    :param save_folder_result: Where to save result of evaluation
    :param data4: test data set 4
    :return: result_df
    '''
    result_df = pd.DataFrame(columns=["test", "model", "window_size_sec", "accuracy", "precision", "recall", "f1"])
    default_param = {
        'max_depth': range(2, 10, 1),
        'n_estimators': range(100, 1000, 100),
        'learning_rate': [0.1, 0.01, 0.05]
    }

    for i in [0.25, 0.5, 0.75, 1,2]:#    for i in [:#

        for df in traindatalist:
            Classification_pipe_all (i, f"{traindatalocation}/{df}" , evaluation ="recall", parameter = default_param , window_size_eval = False, gridsearch = True, save_folder = save_folder_model)
            eval1 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav",data ,i)
            print(eval1)

            result_df = pd.concat([result_df, pd.DataFrame(eval1, index=[0])])

            print(result_df)
            eval2 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav", data2, i)
            result_df = pd.concat([result_df, pd.DataFrame(eval2, index=[0] )])
            print(result_df)
            eval3 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav", data3, i)
            result_df = pd.concat([result_df, pd.DataFrame(eval3, index=[0] )])
            if data4:
                eval4 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav", data4, i)
                result_df = pd.concat([result_df, pd.DataFrame(eval4, index=[0])])
            print(result_df)




    result_df.to_csv(Path.cwd() / save_folder_result)


    return result_df

def eval_all_script_2(data,data2,data3, traindatalist, traindatalocation ="data", save_folder_model =  "data/models ", save_folder_result =  "data/results" ,data4=None ):
    result_df = pd.DataFrame(columns=["test", "model", "window_size_sec", "accuracy", "precision", "recall", "f1"])
    default_param = {
        'max_depth': range(2, 10, 1),
        'n_estimators': range(100, 1000, 100),
        'learning_rate': [0.1, 0.01, 0.05]
    }

    for i in [5]: #0.25, 0.5, 0.75, 1,2
        for df in traindatalist:
            Classification_pipe_all (i, f"{traindatalocation}/{df}" , evaluation ="recall", parameter = default_param , window_size_eval = False, gridsearch = True, save_folder = save_folder_model)
            eval1 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav",data ,i)
            print(eval1)

            result_df = pd.concat([result_df, pd.DataFrame(eval1, index=[0])])

            print(result_df)
            eval2 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav", data2, i)
            result_df = pd.concat([result_df, pd.DataFrame(eval2, index=[0] )])
            print(result_df)
            eval3 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav", data3, i)
            result_df = pd.concat([result_df, pd.DataFrame(eval3, index=[0] )])
            if data4:
                eval4 = evaluation_save(f"{save_folder_model}/model_{df}_grid_{i}.sav", data4, i)
                result_df = pd.concat([result_df, pd.DataFrame(eval4, index=[0])])
            result_df.to_csv(Path.cwd() / save_folder_result)
            print(result_df)




    result_df.to_csv(Path.cwd() / save_folder_result)


    return result_df




if __name__ == "__main__":
    ## testing around##############################################################


    #evaluation("model_oil_pump_small.sav", "data/test_data_2",1)

    # evaluation("model_oil_teflon.sav", "data/test_data_2",1)
    # evaluation("model_oil_vacuum_teflon.sav", "data/test_data_2",1)
    #
    # evaluation("model_oil_pump_small_grid.sav", "data/test_data_2",1)
    #evaluation("model_oil_vacuum_grid_rec.sav", "data/test_data_2",1)
    # evaluation("model_oil_teflon_grid.sav", "data/test_data_2",1)
    #evaluation("model_train_oil_grid.sav", "data/test_data_2",1)
    #


    #evaluation("model_train_oil_vacuum_teflon_grid.sav", "data/test_data_2", 1)

    #evaluation("model_train_oil_vacuum_grid.sav", "data/test_oil_vacuum_teflon",1)
    #evaluation("model_oil_vacuum_grid_acc.sav", "data/test_data_2",1)


    #evaluation("model_train_oil_grid.sav", "data/test_oil_vacuum", 1)

    #evaluation("model_oil_pump_small_grid.sav", "data/Testing_set_oil_vacuum_teflon", 1)
    #evaluation("model_oil_vacuum_grid.sav", "data/Testing_set_oil_vacuum_teflon", 1)


    # important final
    #evaluation("model_electrochem_train_test_grid.sav", "data/electrochem_train_test/val", 1)
    # training = ["train_oil", "train_oil_vacuum", "train_oil_vacuum_teflon"]
    #
    #
    # eval_all_script( "data/test_oil_vacuum", "data/test_oil_vacuum_teflon",
    #                  "data/test_data_2",  traindatalist = training , save_folder_model= "data/models/standard",
    #                  save_folder_result =  "data/results/standard_results.csv")
    #
    #
    # training = ["train_oil_electro", "train_oil_vacuum_electro", "train_oil_vacuum_teflon_electro"]
    # eval_all_script( "data/electro_noise_train/test_oil_vacuum", "data/electro_noise_train/test_oil_vacuum_teflon",
    #                  "data/test_data_2", data4 = "data/electro_noise_train/test_oil_vacuum_teflon_electro",
    #                  traindatalist=training,
    #                 save_folder_model= "data/models/electro_noise", save_folder_result = "data/results/electro_noise_results.csv")
    ## evaluating single
    evaluation("models/standard/model_train_oil_vacuum_grid_1.sav", "data/test_data_2", 1) #classifies all as 1, when adding vacuum and/or teflon, predicts almost everything as 1 but classifies all non cavitation as 1
    # evaluation("models/standard/model_train_oil_vacuum_teflon_grid_0.75.sav", "data/test_data_2", 0.75) #predicts almost everything as 1 but classifies all non cavitation as 1
    # evaluation("models/standard/model_train_oil_vacuum_grid_0.75.sav", "data/test_data_2", 0.75) #predicts almost everything as 1 but classifies all non cavitation as 1
    # evaluation("models/standard/model_train_oil_vacuum_grid_0.5.sav", "data/test_data_2", 0.5) #predicts almost everything as 1 but classifies all non cavitation as 1
    # evaluation("models/standard/model_train_oil_vacuum_grid_0.25.sav", "data/test_data_2", 0.25) #predicts almost everything as 1 but classifies all non cavitation as 1
    #

    ## evaluating own test data
    # evaluation("models/standard/model_train_oil_grid_1.sav", "data/test_oil_vacuum_teflon", 1) # only classified oil pump cavitation as cavitation
    # evaluation("models/standard/model_train_oil_vacuum_grid_1.sav", "data/test_oil_vacuum_teflon", 1) # was able to classify teflon 100%
    # evaluation("models/standard/model_train_oil_vacuum_teflon_grid_1.sav", "data/test_oil_vacuum_teflon", 1) #
    #

    #electrochem eval

    # evaluation("electrochem_1sec/model_electrochem_1sec_grid_1.sav", "data/electrochem_1sec/balanced", 1)
    # evaluation("electrochem_1sec/model_electrochem_1sec_grid_1.sav", "data/electrochem_1sec/imbalanced", 1)
    # evaluation("electrochem_1sec/model_electrochem_1sec_grid_1.sav", "data/electrochem_1sec/superimbalanced", 1)
    # evaluation("electrochem_1sec/50_50/model_electrochem_1sec_grid_1.sav", "data/electrochem_1sec/50_50/balanced", 1)
    # evaluation("electrochem_1sec/50_50/model_electrochem_1sec_grid_1.sav", "data/electrochem_1sec/50_50/imbalanced", 1)
    # evaluation("electrochem_1sec/50_50/model_electrochem_1sec_grid_1.sav", "data/electrochem_1sec/50_50/superimbalanced", 1)
    #
    #evaluation("train_oil_vacuum_add_old_pump/model_train_oil_vacuum_add_old_pump_grid_1.sav", "data/test_data_2", 1)
