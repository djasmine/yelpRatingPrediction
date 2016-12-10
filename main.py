import json
import itertools
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


def get_data():
    print("loading data...")
    review_path = "/Users/JiamingDong/Downloads/yelp/yelp_academic_dataset_review.json"
    file = open(review_path, "r")
    rev_by_star = [[] for i in range(5)]  # use 0 to 4 for star 1 to 5
    for line in file:
        review = json.loads(line)
        # print(review["stars"])
        rev_by_star[review["stars"] - 1].append(review["text"])
    for i in rev_by_star:
        print(len(i))
    print("data loaded")
    return rev_by_star


def split_data(rev_by_star):
    print("start to split data...")
    X_target = []
    Y_target = []
    train_sent = []
    test_sent = []
    for i in range(len(rev_by_star)):
        tot_len = len(rev_by_star[i])
        train_len = int(tot_len * 0.8)  # 80% of training data
        train_sent.extend(rev_by_star[i][:train_len])
        test_sent.extend(rev_by_star[i][train_len:])
        X_target.extend([i] * train_len)
        Y_target.extend([i] * (len(rev_by_star[i]) - train_len))
    print("start to extract feature...")
    tot_train_len = len(train_sent)
    # tot_test_len = len(test_sent)
    train_sent.extend(test_sent)
    del test_sent
    count_vec = CountVectorizer()
    t_count_vec = CountVectorizer(analyzer="word",
                                  tokenizer=None,
                                  preprocessor=None,
                                  ngram_range=(1, 1),
                                  strip_accents='unicode',
                                  max_features=1000)
    X_data = t_count_vec.fit_transform(train_sent)
    print("start to calculate tf-idf")
    print(X_data.shape)
    tfidf_transformer = TfidfTransformer().fit(X_data)
    X_data_tfidf = tfidf_transformer.transform(X_data)
    X_train = X_data_tfidf[:tot_train_len]
    Y_test = X_data_tfidf[tot_train_len:]
    print("data splited")
    print(X_train.shape, Y_test.shape)
    return X_train, Y_test, X_target, Y_target


def plot_confusion_matrix(Y_pred, Y_target, title):
    cnf_matrix = confusion_matrix(Y_target, Y_pred)
    cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cnf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title + " confusion matrix")
    plt.colorbar()
    print(cnf_matrix)
    class_names = [1, 2, 3, 4, 5]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    thresh = cnf_matrix.max() / 2.0
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, "{0:.4f}".format(cnf_matrix[i, j]), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("fig/" + title + ".png")
    print(precision_recall_fscore_support(Y_target, Y_pred, average="macro"))
    print(precision_recall_fscore_support(Y_target, Y_pred, average="micro"))
    print(precision_recall_fscore_support(Y_target, Y_pred, average="weighted"))
    print(mean_squared_error(Y_target, Y_pred))


def test_svm(X_train, Y_test, X_target, Y_target):
    print("start to test SVM")
    t_svm = svm.LinearSVC(C=1.0)
    t_svm.fit(X_train, X_target)
    print("training done.")
    Y_pred = t_svm.predict(Y_test)
    print("Accuracy is : %.2f" % ((Y_target == Y_pred).sum() * 1.0 / (1.0 * Y_test.shape[0])))
    #plot_confusion_matrix(Y_pred, Y_target, "SVM")


def test_multinomial_bayes(X_train, Y_test, X_target, Y_target):
    print("start to test multinomial bayes")
    mnb = MultinomialNB()
    Y_pred = mnb.fit(X_train, X_target).predict(Y_test)
    print("Accuracy is : %.2f" % ((Y_target == Y_pred).sum() * 1.0 / (1.0 * Y_test.shape[0])))
    plot_confusion_matrix(Y_pred, Y_target, "Multinomial Naive Bayes")


def test_bernoulli_bayes(X_train, Y_test, X_target, Y_target):
    print("start to test bernouli bayes")
    bnb = BernoulliNB()
    Y_pred = bnb.fit(X_train, X_target).predict(Y_test)
    print("Accuracy is : %.2f" % ((Y_target == Y_pred).sum() * 1.0 / (1.0 * Y_test.shape[0])))
    plot_confusion_matrix(Y_pred, Y_target, "Bernoulli Naive Bayes")


def test_random_forest(X_train, Y_test, X_target, Y_target):
    print("start to test random forest")
    rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=6)
    rf.fit(X_train, X_target)
    print("training done.")
    Y_pred = rf.predict(Y_test)
    print("Accuracy is : %.2f" % ((Y_target == Y_pred).sum() * 1.0 / (1.0 * Y_test.shape[0])))
    plot_confusion_matrix(Y_pred, Y_target, "random_forest")


def dim_reduction(X_train, Y_test):
    print("start to dimention reduction...")
    t_svd = TruncatedSVD(n_components=200, n_iter=10)
    t_svd.fit(X_train)
    ret_X = t_svd.transform(X_train)
    ret_Y = t_svd.transform(Y_test)
    print("reduction done.")
    return ret_X, ret_Y


def main():
    print("start")
    np.set_printoptions(precision=4, suppress=True)
    rev_by_star = get_data()
    X_train, Y_test, X_target, Y_target = split_data(rev_by_star)
    #X_train, Y_test = dim_reduction(X_train, Y_test)
    #test_multinomial_bayes(X_train, Y_test, X_target, Y_target)
    #test_bernoulli_bayes(X_train, Y_test, X_target, Y_target)
    #test_random_forest(X_train, Y_test, X_target, Y_target)
    test_svm(X_train, Y_test, X_target, Y_target)
    print("end")


if __name__ == "__main__":
    main()
