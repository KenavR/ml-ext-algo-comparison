from statistics import mean, pstdev


def printEvaluationHeader(headline, name):
    print('========================================== ' + headline + ' ==========================================')
    print('')
    print "{:32s} {:10s} {:10s} {:10s} {:15s} {:15s}".format(name, "Accuracy", "Precision", "Recall", "Training time (ms)", "Testing time (ms)")
    print('----------------------------------------------------------------------------------------------')
    return


def printEvaluationRow(classifier, acc, precision, recall, time_train, time_test):
    print "{:30s} {:10.3f} {:10.3f} {:10.3f} {:15.3f} {:15.3f}".format(classifier + " (mean)", mean(acc), mean(precision), mean(recall), mean(time_train), mean(time_test))
    print "{:30s} {:10.3f} {:10.3f} {:10.3f} {:15.3f} {:15.3f}".format(classifier + " (std)", pstdev(acc), pstdev(precision), pstdev(recall),
          pstdev(time_train), pstdev(time_test))
    print('----------------------------------------------------------------------------------------------')
    return