from scripts.svm import train_all, eval_auto
import numpy as np
import pandas as pd
from pathlib import Path




if __name__ == "__main__":
    ## poly and test on senzoro data
    #train window sizes on all data sets
    for i in [0.5, 1, 2]:
        train_all(folders=["data/train_oil", "data/train_oil_vacuum", "data/train_oil_vacuum_teflon"], window_size= i, kernel="poly")
    #evaluate all test_data_2 is senzoro data set
    poly_result = eval_auto(window_size = [0.5, 1, 2], kernel="poly", test_set="test_data_2")


    #
    # #train sigmoid svm and test on senzoro data
    for i in [0.5, 1, 2]:
        train_all(folders=["data/train_oil", "data/train_oil_vacuum", "data/train_oil_vacuum_teflon"], window_size= i, kernel="sigmoid")
    sigmoid_result =  eval_auto(window_size=[0.5, 1, 2], kernel="sigmoid", test_set="test_data_2")
    result = pd.concat([poly_result, pd.DataFrame(sigmoid_result)])



    # #train rbf svm and test on senzoro data
    for i in [0.5, 1, 2]:
        train_all(folders=["data/train_oil", "data/train_oil_vacuum", "data/train_oil_vacuum_teflon"], window_size= i, kernel="rbf")
    rbf_result = eval_auto(window_size=[0.5, 1, 2], kernel="rbf", test_set="test_data_2")
    result = pd.concat([result, pd.DataFrame(rbf_result)])


    #


    # ## Trainig done, additional testing on other test sets from experiments
    #
    # #Outputes performance of all models with own test sets
    kernels = ["poly", "sigmoid", "rbf"]
    for i in kernels:
        test_oil_result = eval_auto(window_size=[0.5, 1, 2], kernel=i, test_set="test_oil")
        result = pd.concat([result, pd.DataFrame(test_oil_result)])

        test_oil_vacuum_result = eval_auto(window_size=[0.5, 1, 2], kernel=i, test_set="test_oil_vacuum")
        result = pd.concat([result, pd.DataFrame(test_oil_vacuum_result)])

        test_oil_vacuum_teflon_result = eval_auto(window_size=[0.5, 1, 2], kernel=i, test_set="test_oil_vacuum_teflon")
        result = pd.concat([result, pd.DataFrame(test_oil_vacuum_teflon_result)])

    result.to_csv(Path.cwd() / f"data/svm_models/final_result_all_models.csv")