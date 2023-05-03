import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from keras import Sequential
from keras.layers import Dense
import pickle

df = pd.read_csv("Recidivism_Challenge.csv")

df = df.dropna()

df = df.drop('ID', axis = 1)

pre_y = df['Recidivism_Within_3years']
pre_X = df.drop('Recidivism_Within_3years', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(pre_X, pre_y, test_size = 0.2, random_state = 42, shuffle = True)

#classifier.fit(X_train, y_train)
classifier = Sequential()
classifier.add(keras.Input(shape =(30,)))
classifier.add(Dense(400, activation = 'relu', kernel_initializer = 'random_normal', input_dim = 30))
classifier.add(Dense(800, activation = 'relu', kernel_initializer = 'random_normal'))
classifier.add(Dense(10, activation = 'relu', kernel_initializer = 'random_normal'))
classifier.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'random_normal'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 20, epochs = 50)

pickle.dump(classifier, open('model.pkl', 'wb'))
