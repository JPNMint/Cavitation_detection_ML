from scripts.svm import *



def svm_classifier_electro(folder, window_size, kernel = 'poly', save_folder = "data/testing_models"):

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

def train_electro(window_size, kernel="poly"):
    svm_classifier_electro(folder="data/electrochem_1sec", window_size=window_size, kernel=kernel,
                               save_folder=f"data/svm_models_electro/{kernel}")



def eval_svm_electro(folder, modelname, testsetfolder, window_size):

    filename = Path().resolve() / f"{folder}/{modelname}"
    loaded_model = pickle.load(open(filename, 'rb'))
    length, data = feature_pipe(testsetfolder, window_size)
    X = data.loc[:, data.columns != 'Cavitation'].astype('float')
    y = data["Cavitation"]
    predicted_y = loaded_model.predict(X)
    print(y)
    print(predicted_y)

    evaluation = metrics.classification_report(y, predicted_y, output_dict=True)#
    0.print(evaluation)
    print ({"test": testsetfolder, "kernel": folder, "model": modelname, "window_size_sec": window_size,
     "accuracy": evaluation["accuracy"], "precision": evaluation["1"]["precision"], "recall": evaluation["1"]["recall"],
     "f1": evaluation["1"]["f1-score"]})

    return {"test": testsetfolder, "kernel": folder, "model" : modelname,"window_size_sec":window_size,"accuracy": evaluation["accuracy"], "precision":evaluation["1"]["precision"],"recall":evaluation["1"]["recall"], "f1":evaluation["1"]["f1-score"]}



if __name__ == "__main__":
    # train_electro(1,kernel= "poly")
    # train_electro(1,kernel= "sigmoid")
    # # train_electro(1,kernel= "rbf")
    # eval_svm_electro(folder = "data/svm_models_electro/poly/" , modelname = "model_electrochem_1sec_svm_poly_1.sav" , testsetfolder = "data/electrochem_1sec/balanced", window_size = 1)
    # eval_svm_electro(folder = "data/svm_models_electro/sigmoid/" , modelname = "model_electrochem_1sec_svm_sigmoid_1.sav", testsetfolder = "data/electrochem_1sec/balanced", window_size = 1)
    # eval_svm_electro(folder = "data/svm_models_electro/rbf/" , modelname = "model_electrochem_1sec_svm_rbf_1.sav", testsetfolder = "data/electrochem_1sec/balanced", window_size = 1)

    #
    # eval_svm_electro(folder = "data/svm_models_electro/poly/" , modelname = "model_electrochem_1sec_svm_poly_1.sav" , testsetfolder = "data/electrochem_1sec/imbalanced", window_size = 1)
    # eval_svm_electro(folder = "data/svm_models_electro/sigmoid/" , modelname = "model_electrochem_1sec_svm_sigmoid_1.sav", testsetfolder = "data/electrochem_1sec/imbalanced", window_size = 1)
    # eval_svm_electro(folder = "data/svm_models_electro/rbf/" , modelname = "model_electrochem_1sec_svm_rbf_1.sav", testsetfolder = "data/electrochem_1sec/imbalanced", window_size = 1)

    eval_svm_electro(folder = "data/svm_models_electro/poly/" , modelname = "model_electrochem_1sec_svm_poly_1.sav" , testsetfolder = "data/electrochem_1sec/superimbalanced", window_size = 1)
    eval_svm_electro(folder = "data/svm_models_electro/sigmoid/" , modelname = "model_electrochem_1sec_svm_sigmoid_1.sav", testsetfolder = "data/electrochem_1sec/superimbalanced", window_size = 1)
    eval_svm_electro(folder = "data/svm_models_electro/rbf/" , modelname = "model_electrochem_1sec_svm_rbf_1.sav", testsetfolder = "data/electrochem_1sec/superimbalanced", window_size = 1)


    #svm_models_electro
    ## poly and test

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