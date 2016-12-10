from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from main import get_data, split_data, plot_confusion_matrix
from keras.utils.np_utils import to_categorical, categorical_probas_to_classes
import numpy as np


def build_model(input_num, output_num):
    model = Sequential([
        Dense(300, input_dim=input_num),
        Activation("relu"),
        Dropout(0.3),
        Dense(100),
        Activation("relu"),
        Dropout(0.2),
        Dense(output_num),
        Activation('softmax'),
    ])
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def trans_target(target):
    ret = np.zeros((target.shape[0], 5), dtype=np.int)
    for i in range(target.shape[0]):
        ret[i, target[i]] = 1
    print(ret.shape)
    print(ret[0])
    return ret


def main():
    rev_by_star = get_data()
    X_train, Y_test, X_target, Y_target = split_data(rev_by_star)
    X_train = X_train.toarray()
    Y_test = Y_test.toarray()
    #Y_target = trans_target(Y_target)
    #X_target = trans_target(X_target)
    input_num = 1000
    output_num = 5
    X_target = to_categorical(X_target, 5)
    #Y_target = to_categorical(Y_target, 5)
    #data = np.random.random((2148051, input_num))
    #labels = np.random.randint(output_num, size=(2148051, 1))
    #print(X_target[0])
    #print(X_train.dtype, X_target.dtype)
    #labels = to_categorical(labels, 5)
    #print("data shape", data.dtype)
    #print("label shape", labels.dtype)
    print(type(X_train))
    model = build_model(input_num, output_num)
    model.fit(X_train, X_target, batch_size=128, nb_epoch=5, validation_split=0.25)
    #model.fit(data, labels, batch_size=32, nb_epoch=10)
    Y_pred = model.predict(Y_test)
    Y_pred = categorical_probas_to_classes(Y_pred)
    print("Accuracy is : %.2f" % ((Y_target == Y_pred).sum() * 1.0 / (1.0 * Y_test.shape[0])))
    plot_confusion_matrix(Y_pred, Y_target, "neural_network")


if __name__ == "__main__":
    main()
