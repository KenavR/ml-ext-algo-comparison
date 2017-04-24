from sklearn import metrics

def convertMetrics(y_test, y_test_predicted, start_time_train, end_time_train, start_time_test, end_time_test):
    acc = metrics.accuracy_score(y_test, y_test_predicted)
    precision = metrics.precision_score(y_test, y_test_predicted, average="micro")
    recall = metrics.recall_score(y_test, y_test_predicted, average="micro")
    time_train = round((end_time_train - start_time_train) * 1000, 3)
    time_test = round((end_time_test - start_time_test) * 1000, 3)
    confusion_matrix = metrics.confusion_matrix(y_test, y_test_predicted)

    return [acc, precision, recall, time_train, time_test, confusion_matrix]