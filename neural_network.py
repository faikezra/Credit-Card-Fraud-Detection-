import random
import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def Neural_Network_Selection(normal_count, anomaly_coount, data_dir):
    undersample , _ = preprocessing.create_datasets(data_dir, normal_count, anomaly_coount)
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = undersample
    hidden_layers = [1,2,3,4,5]
    hidden_layer_neurons = [50,100,200,300,500]
    results_matrix_validation = np.zeros((5,5))
    dense_index = -1

    for neurons in hidden_layer_neurons:
        layer_index = -1
        dense_index += 1
        for layers in hidden_layers:
            model = Sequential()
            model.add(Dense(neurons, input_dim = 30, activation='relu'))
            layer_index += 1
            for _layers in range (layers):
                model.add(Dense(neurons, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            avg_acc = 0
            # Fit the model
            model.fit(X_train_undersample, y_train_undersample, epochs=200, batch_size=10,verbose=0)
            acc = (model.evaluate(X_test_undersample,y_test_undersample)[1])
            results_matrix_validation[dense_index][layer_index] = acc
            #print('accuracy for a neural network with {} hidden layers, and {} dense connections is: {}'.format(layers, neurons, acc))

    result = np.where(results_matrix_validation == np.amax(results_matrix_validation))
    #print("------------------------------------")
    #print("Best Validation Neural Network Acc: {}".format(np.amax(results_matrix_validation)))
    #print("With Dense Connection Number: {}".format(hidden_layer_neurons[result[0][0]]))
    #print("With Hidden Layer Depth: {}".format(hidden_layers[result[1][0]]))
    #print("------------------------------------")
    return (hidden_layer_neurons[result[0][0]], hidden_layers[result[1][0]])

def main():
    undersample_amount = 200
    data_dir = '/Users/ezra/Documents/data_repo/creditcard.csv'
    return Neural_Network_Selection(undersample_amount, undersample_amount, data_dir)

if __name__ == "__main__":
    main()
