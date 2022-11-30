import pickle
from pathlib import Path
from scripts.utils import feature_pipe
from sklearn import metrics
import numpy as np
import pandas as pd

def predict(model_name, testset_folder, window_size):
    path = Path.cwd()/"data"/model_name
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



if __name__ == "__main__":
    predict ("SVM_model_no_grid.sav", "data/trainingset_vacuum_senzoro",1)
