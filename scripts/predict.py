from svm import *
from utils import read_wav, feature_extraction, splitting, fourier_trans


def predict(model, to_predict, window_size):
    print(f"Predicting {to_predict} with model {model}!")
    filename = Path().resolve() / f"{model}"
    loaded_model = pickle.load(open(filename, 'rb'))
    length, X = feature_pipe_predict(to_predict, window_size)
    #X = data.loc[:, data.columns != 'Cavitation'].astype('float')
    #y = data["Cavitation"]
    predicted_y = loaded_model.predict(X)
    print(predicted_y)






def feature_pipe_predict (path, splits_in_sec):
    target_path = Path.cwd()/path
    print(f"Extracting features from {target_path}")
    file_names = list((target_path).glob('*.wav'))

    #read files
    sr, df = read_wav(file_names)
    #split files
    df_split = splitting(sr[0],df, splits_in_sec)
    #fourier trans
    freq, df_ftt = fourier_trans(sr[0],df_split)
    #extract features
    output_df = feature_extraction(df_ftt, None, sr = sr[0]) #cavitation = 1
    return len(df_split), output_df


    # sr_no_cav, df_no_cav = read_wav(no_cav_f_names)
    # df_split_no_cav = splitting(sr_no_cav[0], df_no_cav, splits_in_sec)
    # freq_no_cav, df_ftt_no_cav = fourier_trans(sr_no_cav[0], df_split_no_cav)
    # output_df_no_cav = feature_extraction(df_ftt_no_cav, "No Cavitation", sr = sr_no_cav[0]) #no_cavitation = 0
    #
    #
    # df = pd.concat([output_df_cav, output_df_no_cav])
    # print(df)
    # return len(df_split_cav)+len(df_split_no_cav), df



if __name__ == "__main__":
    #feature_pipe_predict("data/Senzoro_blind_data",1)
    ## sigmoid
    # predict("data/svm_models/sigmoid/model_train_oil_svm_sigmoid_1.sav", "data/Senzoro_blind_data", 1)
    # predict("data/svm_models/sigmoid/model_train_oil_vacuum_svm_sigmoid_1.sav", "data/Senzoro_blind_data",1)
    # predict("data/svm_models/sigmoid/model_train_oil_vacuum_teflon_svm_sigmoid_1.sav", "data/Senzoro_blind_data", 1)


    ##poly
    # predict("data/svm_models/poly/model_train_oil_svm_poly_1.sav", "data/Senzoro_blind_data", 1)
    # predict("data/svm_models/poly/model_train_oil_vacuum_svm_poly_1.sav", "data/Senzoro_blind_data",1)
    # predict("data/svm_models/poly/model_train_oil_vacuum_teflon_svm_poly_1.sav", "data/Senzoro_blind_data", 1)


    #
    ## rbf
    # predict("data/svm_models/rbf/model_train_oil_svm_rbf_1.sav", "data/Senzoro_blind_data", 1)
    # predict("data/svm_models/rbf/model_train_oil_vacuum_svm_rbf_1.sav", "data/Senzoro_blind_data", 1)
    # predict("data/svm_models/rbf/model_train_oil_vacuum_teflon_svm_rbf_1.sav", "data/Senzoro_blind_data", 1)

