'''
ニューラルネットワークの特徴量を比較する関数群
'''

# data analisis
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# neural network
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn.preprocessing import StandardScaler
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam

# 荷重減衰の比較
def compare_weight_decay(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    acc = []
    decay = np.linspace(0, 0.003, 100)
    for i in decay:
        adam = keras.optimizers.Adam(decay=i)
        model.compile(loss='categorical_crossentropy', optimizer=adam,\
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train_onehot, batch_size=10, epochs=50,\
                            verbose=0, validation_data=(x_test, y_test_onehot))

        y_prediction = model.predict_classes(x_test)

        print(np.sum(y_prediction == y_test) / float(len(y_test)))
        acc.append(np.sum(y_prediction == y_test) / float(len(y_test)))

    plt.plot(decay, acc)
    plt.title('validation accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('weight decay')
    plt.savefig('../fig/weight_decay.png')
    plt.show()

    exit()

# 層の比較
def compare_layer(x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    acc = []
    layers = np.arange(0, 10, 1)
    # 層の数を変えて比較
    for layer_num in layers:
        model = Sequential()
        model.add(Dense(input_dim=8, units=16, init='he_uniform', activation='relu'))
        for i in range(layer_num): # 同じ形状の層を重ねる
            model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=2, activation='softmax'))

        adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001,\
                                     amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=adam,\
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train_onehot, batch_size=10, epochs=50,\
                            verbose=0, validation_data=(x_test, y_test_onehot))

        y_prediction = model.predict_classes(x_test)

        print(np.sum(y_prediction == y_test) / float(len(y_test)))
        acc.append(np.sum(y_prediction == y_test) / float(len(y_test)))

    plt.plot(layers, acc)
    plt.title('validation accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('number of layer')
    plt.savefig('../fig/layer.png')
    plt.show()

    exit()

# ニューロン数の比較
def compare_units(x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    acc = []
    units = np.arange(1, 500, 1)
    print(units)

    for unit in units:
        model = Sequential()
        model.add(Dense(input_dim=8, units=unit, init='he_uniform', activation='relu'))
        model.add(Dense(units=2, activation='softmax'))

        adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None,\
                                     decay=0.001, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=adam,\
                     metrics=['accuracy'])

        history = model.fit(x_train, y_train_onehot, batch_size=10, epochs=50,\
                            verbose=0, validation_data=(x_test, y_test_onehot))

        y_prediction = model.predict_classes(x_test)

        print(np.sum(y_prediction == y_test) / float(len(y_test)))
        acc.append(np.sum(y_prediction == y_test) / float(len(y_test)))

    plt.plot(units, acc)
    plt.title('validation accuracy')
    plt.ylabel('validation accuracy')
    plt.xlabel('number of unit')
    plt.savefig('../fig/unit.png')
    plt.show()

    exit()

# 最適化関数の比較
def compare_optimizer(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    optimizers = [SGD, Adadelta, Adamax, Adam, Adagrad,  RMSprop, Nadam]
    histories = {}

    for optimizer in optimizers:
        model = Sequential()
        model.add(Dense(input_dim=8, units=32, init='he_uniform', activation='relu'))
        model.add(Dense(units=2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer(),\
                      metrics=['accuracy'])
        histories[optimizer.__name__] = model.fit(x_train, y_train_onehot, batch_size=10, epochs=50,\
                            verbose=0, validation_data=(x_test, y_test_onehot))

        y_prediction = model.predict_classes(x_test)

    # x = range(10)
    # plot accuracy of train data
    # for k, result in histories.items():
    #     plt.plot(x, result.history['acc'], label=k)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plot accuracy of validation data
    # for k, result in histories.items():
    #     plt.plot(x, result.history['val_acc'], label=k)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.savefig('../fig/opt.png')
    # plt.show()

    exit()

# 重みの初期値の比較
def compare_init(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    exit()

# 活性化関数の比較
def compare_activation(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    exit()

# batch normalizationの比較
def compare_batch_normalization(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    exit()

# Dropoutの比較
def compare_dropout(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    exit()

# 学習係数の比較
def compare_learning_rate(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    exit()

# バッチサイズの比較
def compare_batch_size(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    exit()

# エポック数の比較
def compare_epochs(model, x_train, y_train_onehot, x_test, y_test, y_test_onehot):
    exit()

