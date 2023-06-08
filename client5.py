import pandas as pd
import numpy as np
from numpy import *
from keras.layers import MaxPooling1D
from keras.layers import Dense 

from tensorflow import keras

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D,Conv1D, MaxPool2D, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score


from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D 
from tensorflow.keras import layers


import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np



data = pd.read_csv('df_5.csv', dtype=object)

where_are_NaNs = pd.isna(data)
data[where_are_NaNs] = 0
training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")

Y_train = training_data["Efficiency"]
X_train = training_data.drop(labels = ["egoid","Efficiency","timetobed","timeoutofbed", "bedtimedur", "minstofallasleep", "minsafterwakeup", "minsasleep", "minsawake", "datadate"],axis = 1) 

Y_test = testing_data["Efficiency"]
X_test = testing_data.drop(labels = ["egoid","Efficiency","timetobed","timeoutofbed", "bedtimedur", "minstofallasleep", "minsafterwakeup", "minsasleep", "minsawake", "datadate"],axis = 1) 

cols = X_train.columns

X_test = np.asarray(X_test).astype(np.float32)
Y_test = np.asarray(Y_test).astype(np.float32)
X_train = np.asarray(X_train).astype(np.float32)
Y_train = np.asarray(Y_train).astype(np.float32)



model = Sequential()
model.add(Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(82,1)))
model.add(MaxPooling1D(
    pool_size=2,
    strides=None,
    padding='valid',
    data_format='channels_last'
))

model.add(Dense(16, input_shape = (82,1)))
model.add(Dense(16))

model.add(Flatten())

model.compile(optimizer=Adam(learning_rate=0.5), loss='mean_squared_error', metrics=['mse'])



# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test), verbose=0)
        hist = r.history
        print("Fit history : " ,hist)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
        print("Eval accuracy : ", accuracy)
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient(), 
        grpc_max_message_length = 1024*1024*1024
)