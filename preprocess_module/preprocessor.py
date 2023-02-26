from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os 
import pickle
def label_encod(data,column_name):
    main_path = os.path.join("Dataset","artifacts")
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    encoder_path = os.path.join("Dataset","artifacts",column_name+".pkl")
    if not os.path.exists(encoder_path): 
        encoder = LabelEncoder()
        transformed_data = encoder.fit_transform(data)
        pickle.dump(encoder,open(encoder_path, 'wb'))
    else: # If it's their, then I just want to you use it to inverse transform // Could be used to transform but no need to make several functions as in this project type we only transform the original data and won't use it again
        encoder = pickle.load(open(encoder_path,'rb')) 
        transformed_data = encoder.inverse_transform(data)
    return transformed_data

def scale_data(data,column_name):
    main_path = os.path.join("Dataset","artifacts")
    if not os.path.exists(main_path):
        os.makedirs(main_path)
        
    scaler_path = os.path.join("Dataset","artifacts",column_name+".pkl")
    if not os.path.exists(scaler_path):
        scaler = MinMaxScaler()
        transformed_data = scaler.fit_transform(data.to_numpy().reshape(-1,1))
        pickle.dump(scaler,open(scaler_path, 'wb'))
    else:
        scaler = pickle.load(open(scaler_path,'rb')) 
        transformed_data = scaler.inverse_transform(data.to_numpy().reshape(-1,1))
    return transformed_data
