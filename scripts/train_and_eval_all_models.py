from scripts.svm import train_all






if __name__ == "__main__":
    ## poly and test on senzoro data
    #train window sizes on all data sets
    for i in [0.5, 1, 2]:
        train_all(folders=["data/train_oil", "data/train_oil_vacuum", "data/train_oil_vacuum_teflon"], window_size= i, kernel="poly")
    #evaluate all test_data_2 is senzoro data set
    eval_auto(window_size = [0.5, 1, 2], kernel="poly", test_set="test_data_2")

    #eval_auto(kernel = "poly", test_set = "test_data_2")


    # # ## train sigmoid and test on senzoro data
    #for i in [0.5, 1, 2]:
    #    train_all(folders=["data/train_oil", "data/train_oil_vacuum", "data/train_oil_vacuum_teflon"], window_size= i, kernel="sigmoid")
    #eval_auto(window_size=[0.5, 1, 2], kernel="sigmoid", test_set="test_data_2")

    #for i in [0.5, 1, 2]:
    #    train_all(folders=["data/train_oil", "data/train_oil_vacuum", "data/train_oil_vacuum_teflon"], window_size= i, kernel="rbf")
    #eval_auto(window_size=[0.5, 1, 2], kernel="rbf", test_set="test_data_2")




    ## Trainig done, additional testing on other test sets from experiments

    #Outputes performance of all models with own test sets
    kernels = ["poly", "sigmoid", "rbf"]
    for i in kernels:
        eval_auto(window_size=[0.5, 1, 2], kernel=i, test_set="test_oil")
        eval_auto(window_size=[0.5, 1, 2], kernel=i, test_set="test_oil_vacuum")
        eval_auto(window_size=[0.5, 1, 2], kernel=i, test_set="test_oil_vacuum_teflon")


