from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scripts.utils import feature_pipe
from sklearn import metrics
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import  GridSearchCV


def svm_classifier(folder, window_size, kernel = 'poly', save_folder = "data/testing_models"):
    '''
    Training pipe for svm
    :param folder: Input data set, should go through feature_pipe function first
    :param window_size: window size
    :param kernel: kernel for svm
    :param save_folder: folder to save model

    '''

    print(f"Training model with {folder} set!")
    length, data = feature_pipe(folder, window_size)
    X = data.loc[:, data.columns != 'Cavitation'].astype('float')
    y = data["Cavitation"]
    print("Done!")


    if kernel == "rbf":
        print(f"Kernel is {kernel}, initiating hypertuning of parameter!")
        C_range = 10. ** np.arange(-3, 8)
        gamma_range = 10. ** np.arange(-5, 4)

        #param_grid = dict(gamma=gamma_range, C=C_range)

        param_grid = {
            'kernel': ['rbf'],
            'C' : C_range,
            'gamma' : gamma_range
            #'C': [0.1, 1.0, 100],
            #'gamma': [0, 1, 10]
        }
        print(f"Fitting models! Window size is set to {window_size} second(s)")
        svm = SVC(kernel=kernel, random_state=1623806)
        classifier = GridSearchCV(estimator=svm, param_grid=param_grid,
                             verbose=10, n_jobs=4, cv=5) #, scoring=ftwo_scorer
        classifier.fit(X, y)
        print("GridSearch done!")
        print("Best score: %0.3f" % classifier.best_score_)
        print("Best Parameters set:")
        best_parameters = classifier.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    else:
        print(f"Kernel is {kernel}!")
        print(f"Fitting model! Window size is set to {window_size} second(s)")
        classifier = SVC(kernel=kernel, random_state=1623806)


        classifier.fit(X, y)
        print("Training done!")
        # save model

    filename = Path().resolve() / f"{save_folder}/model_{Path(folder).parts[1]}_svm_{kernel}_{window_size}.sav"

    pickle.dump(classifier, open(filename, 'wb'))

    print(f"Model saved in {save_folder} as model_{Path(folder).parts[1]}_svm_{kernel}_{window_size}.sav! ")

def eval_svm(folder, modelname, testsetfolder, window_size):
    '''
    Evaluation pipe for svm
    :param folder: folder of model
    :param modelname: model to load
    :param testsetfolder: folder of test set
    :param window_size: window size
    :return: dict of result


    '''
    print(f"Evaluating model {modelname} with test set {testsetfolder}!")
    filename = Path().resolve() / f"{folder}/{modelname}"
    loaded_model = pickle.load(open(filename, 'rb'))
    length, data = feature_pipe(testsetfolder, window_size)
    X = data.loc[:, data.columns != 'Cavitation'].astype('float')
    y = data["Cavitation"]
    predicted_y = loaded_model.predict(X)
    print("True values:")
    print(np.asarray(y))
    print("Predicted values:")
    print(predicted_y)

    evaluation = metrics.classification_report(y, predicted_y, output_dict=True)#
    print(f"Accuracy: {evaluation['accuracy']}, Precision on class 1: {evaluation['1']['precision']}")
    print(f"Recall on class 1: {evaluation['1']['recall']}, f1: {evaluation['1']['f1-score']}")

    return {"test": testsetfolder, "kernel": folder, "model" : modelname,"window_size_sec":window_size,"accuracy": evaluation["accuracy"], "precision":evaluation["1"]["precision"],"recall":evaluation["1"]["recall"], "f1":evaluation["1"]["f1-score"]}


def train_all(window_size, kernel = "poly",folders = ["data/train_oil", "data/train_oil_vacuum", "data/train_oil_vacuum_teflon"]):
    '''
    Initiates train of multiple models
    :param: folders: folder names of trainsets as list
    :param window_size: window size
    :param kernel: kernel of svm
    :return: dict of result


    '''
    for i in folders:
        svm_classifier(folder = i, window_size = window_size, kernel = kernel, save_folder = f"data/svm_models/{kernel}")



def eval_all(save_folder_result, window_size ,kernel, test_set = "test_data_2"):
    '''
    Evaluation pipe for svm
    :param: save_folder_result: save folder of results
    :param window_size: window size
    :param kernel: kernel of svm
    :param test_set: test set
    :return: result_df


    '''
    print(f"Evaluating models of kernel {kernel}!")
    print(f"Window size is {window_size}!")
    result_df = pd.DataFrame(columns=["test", "model", "kernel" , "window_size_sec", "accuracy", "precision", "recall", "f1"])
    #first model with train oil set
    eval1 = eval_svm(folder=f"data/svm_models/{kernel}", modelname=f"model_train_oil_svm_{kernel}_{window_size}.sav", testsetfolder=f"data/{test_set}",
             window_size=window_size)


    result_df = pd.concat([result_df, pd.DataFrame(eval1, index=[0])])
    # second model with train oil_vacuum set

    eval2 = eval_svm(folder = f"data/svm_models/{kernel}", modelname = f"model_train_oil_vacuum_svm_{kernel}_{window_size}.sav", testsetfolder = f"data/{test_set}" , window_size = window_size )
    result_df = pd.concat([result_df, pd.DataFrame(eval2, index=[0])])

    # second model with train oil_vacuum_teflon set
    eval3 = eval_svm(folder = f"data/svm_models/{kernel}", modelname = f"model_train_oil_vacuum_teflon_svm_{kernel}_{window_size}.sav", testsetfolder = f"data/{test_set}" , window_size = window_size )
    result_df = pd.concat([result_df, pd.DataFrame(eval3, index=[0])])
    result_df.to_csv(Path.cwd() / save_folder_result)
    print(result_df)
    return result_df

def eval_auto(window_size, kernel = "poly", test_set = "test_data_2"):
    '''
    Initiates evaluation of models, all window sizes, saves results as csv


    :param window_size: window sizes
    :param: test_set: test data
    :param kernel: kernel of svm



    '''

    result_df = eval_all(f"data/svm_models/{kernel}/results_{window_size[0]}_{test_set}.csv", window_size[0], kernel = kernel, test_set = test_set)

    res_2 = eval_all(f"data/svm_models/{kernel}/results_{window_size[2]}_{test_set}.csv", window_size[2], kernel = kernel, test_set = test_set)
    result_df = pd.concat([result_df, pd.DataFrame(res_2)])

    res_1 = eval_all(f"data/svm_models/{kernel}/results_{window_size[2]}_{test_set}.csv", window_size[1], kernel = kernel, test_set = test_set)
    result_df = pd.concat([result_df, pd.DataFrame(res_1)])
    result_df.to_csv(Path.cwd() / f"data/svm_models/{kernel}/results_{test_set}.csv")


    return result_df





#if __name__ == "__main__":


    ## poly and test on senzoro data

    # train_all(window_size=0.5, kernel= "poly")

    # train_all(window_size=1 ,  kernel= "poly")
    #
    # train_all(window_size=2 ,  kernel= "poly")
    #eval_auto(kernel = "poly", test_set = "test_data_2")


    # # ## train sigmoid and test on senzoro data
    # train_all(window_size=0.5, kernel= "sigmoid")
    # # 1
    # train_all(window_size=1 ,  kernel= "sigmoid")
    #
    # train_all(window_size=2 ,  kernel= "sigmoid")
    #eval_auto(kernel = "sigmoid", test_set = "test_data_2")


    # ## train rbf and test on senzoro data
    # train_all(window_size=0.5, kernel= "rbf")
    # # 1
    # train_all(window_size=1 ,  kernel= "rbf")
    #
    # train_all(window_size=2 ,  kernel= "rbf")
    #eval_auto(kernel = "rbf", test_set = "test_data_2")

    ## TRAINING ALL DONE evaluate all on own set and write csv


    # eval_auto(kernel = "poly", test_set = "test_oil")
    # eval_auto(kernel = "sigmoid", test_set = "test_oil")
    # eval_auto(kernel = "rbf", test_set = "test_oil")
    #
    # eval_auto(kernel = "poly", test_set = "test_oil_vacuum")
    # eval_auto(kernel = "sigmoid", test_set = "test_oil_vacuum")
    # eval_auto(kernel = "rbf", test_set = "test_oil_vacuum")
    #
    # eval_auto(kernel = "poly", test_set = "test_oil_vacuum_teflon")
    # eval_auto(kernel = "sigmoid", test_set = "test_oil_vacuum_teflon")
    # eval_auto(kernel = "rbf", test_set = "test_oil_vacuum_teflon")


    # TEST IF SIGMOID IS REALLY THAT GOOD

    #svm_classifier(folder = "data/train_oil_vacuum", window_size = 1, kernel='sigmoid', save_folder="data/testing_models")
    #eval_svm(folder="data/testing_models", modelname="model_train_oil_vacuum_svm_sigmoid_1.sav", testsetfolder=f"data/test_data_2",
    #             window_size=1)