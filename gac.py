#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""Multilayer perceptron."""

import logging
import logging.config
import sys
from operator import itemgetter

import numpy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    import ConfigParser as configparser
except ImportError:
    import configparser


def get_parameter_from_config(section, parameter):
    """
    Get parameter from the config file named config.ini.

    :param section: name of the section in the config file
    :type section: str
    :param parameter: name of the parameter in the section
    :type parameter: str
    :return: the parameter's value
    :rtype: str
    """
    config = configparser.ConfigParser()
    config.read("config.ini")
    try:
        return config.get(section, parameter)
    except:
        logging.error("No such section : "+section
                      + " or parameter : "+parameter)
        sys.exit(1)


def init_logging():
    """Init logging."""
    logconf_file = get_parameter_from_config("FILES", "logging")
    logging.config.fileConfig(logconf_file)


def import_data():
    """
    Import data from file.

    :return: datas
    :rtype: array(array(float, float))
    """
    dat_file = get_parameter_from_config("FILES", "training_data")
    with open(dat_file, 'r') as op_file:
        res = []
        X = []
        y = []
        for one_line in op_file:
            one_data = one_line.rstrip().split(';')
            X.append([float(one_data[0])])
            y.append(float(one_data[1]))
            # res.append(map(float, one_data))
        op_file.close()
    res.append(X)
    res.append(y)
    return res


def build_mlp(hidden_layer, acti, learning, learning_init, max_it):
    """
    Build a MLPClassifier.

    :param hidden_layer: description of the hidden layers
    :type hidden_layer: tuple
    :param acti: function activating neurons
    :type acti: str
    :param learning: type of learning
    :type learning: str
    :param learning_init: initial learning rate
    :type learning_init: float
    :param max_it: max iterations
    :type max_it: int
    :return: mlp
    :rtype: MLPClassifier
    """
    return MLPClassifier(hidden_layer_sizes=hidden_layer,
                         activation=acti, solver='sgd',
                         learning_rate=learning,
                         learning_rate_init=learning_init,
                         max_iter=max_it)


def summarize_res(x_test, y_test, predictions):
    """
    Build a summary of results.

    :param x_test: x values tested
    :type x_test: array
    :param y_test: y values tested
    :type y_test: array
    :param predictions: y values obtained by mlp
    :type predictions: array
    :return: summary, unique y and pred
    """
    dejavu = []
    res = []
    for i, p in enumerate(predictions):
        if x_test[i] not in dejavu:
            dejavu.append(x_test[i])
            res.append((x_test[i], float(y_test[i]), float(p)))
    return sorted(res, key=itemgetter(0))


def compute_accuracy(expected, obtained):
    """
    Compute accuracy between the values expected and obtained.

    :param expected: value expected
    :type expected: float
    :param obtained: value obtained
    :type obtained: float
    :return: accuracy between the two values
    :rtype: float
    """
    ratio = obtained/expected
    if ratio == 1.0:
        return 100.0
    else:
        return 100 - abs(1-ratio)*100.0


def compute_global_accuracy(expecteds, obtaineds):
    """
    Compute global accuracy between the values obtained and expected.

    :param expecteds: values expected
    :type expecteds: array(float)
    :param obtaineds: values obtained
    :type obtaineds: array(float)
    :return: global accuracy
    :rtype: float
    """
    summ = 0
    for x, y in zip(expecteds, obtaineds):
        summ += compute_accuracy(x, y)
    return summ/len(expecteds)


def write_res(summaries, accuracies, avg_accuracy, mlp_params, filename):
    """
    Write result of one mlp.

    :param summaries: all summaries of the different tests for the mlp
    :type summaries: array
    :param accuracies: accuracy for each test
    :type accuracies: list(float)
    :param avg_accuracy: average accuracy of one mlp on all the tests
    :type avg_accuracy: float
    :param mlp_params: mlp parameters
    :type mlp_params: array
    :param filename: name of the file for the save
    :type filename: str
    """
    fullpath = get_parameter_from_config("FILES", "save_path")+filename
    with open(fullpath, 'w+') as opfile:
        logging.info("Writing results")
        opfile.write("Parameters : \n")
        for one_params in mlp_params:
            opfile.write(str(one_params)+" -- ")
        opfile.write("\n \n")
        opfile.write("Avg accuracy : "+str(avg_accuracy)+"\n")
        for i in range(len(summaries)-1):
            for item in summaries[i]:
                opfile.write(str(item)+" -- ")
            opfile.write("\n Accuracy : "+str(accuracies[i])+"\n\n")
        opfile.close()


def test_mlp(mlp, xs, ys, mlp_params, filename):
    """
    Test a mlp and write results in file.

    :param mlp: mlp
    :type mlp: MLPClassifier
    :param xs: x values
    :type xs: array
    :param ys: y values
    :type ys: array
    :param mlp_params: mlp parameters
    :type mlp_params: array
    :param filename: filename
    :type filename: str
    """
    summaries = []
    accuracies = []
    logging.info("Testing")
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(xs, ys,
                                                            random_state=3)
        y_train = numpy.asarray(y_train, dtype="|S6")
        mlp.fit(x_train, y_train)
        predictions = mlp.predict(x_test)
        summ = summarize_res(x_test, y_test, predictions)
        expecteds = [a[1] for a in summ]
        obtaineds = [a[2] for a in summ]
        summaries.append(summ)
        accuracies.append(compute_global_accuracy(expecteds, obtaineds))
    logging.info("Compute accuracy")
    avg_accuracy = sum(accuracies)/len(accuracies)
    write_res(summaries, accuracies, avg_accuracy, mlp_params, filename)


if __name__ == "__main__":
    init_logging()
    logging.info("Importing data")
    datas = import_data()
    ys = datas[1]*50
    xs = datas[0]*50

#    mlp = build_mlp((10,), 'tanh', 'adaptive', 0.2, 500)
#    mlp_params = [(10,), 'tanh', 'adaptive', 0.2, 500]
#    test_mlp(mlp, x, y, mlp_params, "mlp1.txt")
#
#    mlp = build_mlp((5,), 'tanh', 'adaptive', 0.2, 500)
#    mlp_params = [(5,), 'tanh', 'adaptive', 0.2, 500]
#    test_mlp(mlp, x, y, mlp_params, "mlp2.txt")

    mlp = build_mlp((20,), 'tanh', 'adaptive', 0.2, 500)
    mlp_params = [(20,), 'tanh', 'adaptive', 0.2, 500]
    test_mlp(mlp, xs, ys, mlp_params, "mlpbis.txt")

#    mlp = build_mlp((10,), 'logistic', 'adaptive', 0.2, 500)
#    mlp_params = [(10,), 'logistic', 'adaptive', 0.2, 500]
#    test_mlp(mlp, x, y, mlp_params, "mlp4.txt")
#
#    mlp = build_mlp((10,), 'tanh', 'invscaling', 0.2, 500)
#    mlp_params = [(10,), 'tanh', 'invscaling', 0.2, 500]
#    test_mlp(mlp, x, y, mlp_params, "mlp5.txt")
#
#    mlp = build_mlp((10,), 'tanh', 'constant', 0.2, 500)
#    mlp_params = [(10,), 'tanh', 'constant', 0.2, 500]
#    test_mlp(mlp, x, y, mlp_params, "mlp6.txt")
#
#    mlp = build_mlp((10,), 'tanh', 'adaptive', 0.1, 500)
#    mlp_params = [(10,), 'tanh', 'adaptive', 0.1, 500]
#    test_mlp(mlp, x, y, mlp_params, "mlp7.txt")
#
#    mlp = build_mlp((10,), 'tanh', 'adaptive', 0.3, 500)
#    mlp_params = [(10,), 'tanh', 'adaptive', 0.3, 500]
#    test_mlp(mlp, x, y, mlp_params, "mlp8.txt")
#
#    mlp = build_mlp((10,), 'tanh', 'adaptive', 0.2, 300)
#    mlp_params = [(10,), 'tanh', 'adaptive', 0.2, 300]
#    test_mlp(mlp, x, y, mlp_params, "mlp9.txt")
#
#    mlp = build_mlp((10,), 'tanh', 'adaptive', 0.2, 700)
#    mlp_params = [(10,), 'tanh', 'adaptive', 0.2, 700]
#    test_mlp(mlp, x, y, mlp_params, "mlp10.txt")
