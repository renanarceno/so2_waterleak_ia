from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools

import pandas as pd
import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["pessoas", "maquinas", "vazao_total", "vazao_1", "sensor_p1", "vazao_2", "sensor_p2", "vazao_3", "sensor_p3", "vazamento", "t_seg", "hora"]
FEATURES_VAZ = ["pessoas", "maquinas", "vazao_total", "vazao_1", "sensor_p1", "vazao_2", "sensor_p2", "vazao_3", "sensor_p3", "vazamento"]
FEATURES = ["pessoas", "maquinas", "vazao_total", "vazao_1", "sensor_p1", "vazao_2", "sensor_p2", "vazao_3", "sensor_p3"]
LABEL = "vazamento"


def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)


def predict_value(pessoas, maquinas, vazao_total, vazao_1, sensor_p1, vazao_2, sensor_p2, vazao_3, sensor_p3, vazamento):
    # Load datasets
    training_set = pd.read_csv("hidro2_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("hidro2_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

    dados = dict(zip(FEATURES_VAZ, [pessoas, maquinas, vazao_total, vazao_1, sensor_p1, vazao_2, sensor_p2, vazao_3, sensor_p3, vazamento]))
    predict_func = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame([dados], columns=dados.keys()),
        y=pd.Series(dados[LABEL]),
        num_epochs=1,
        shuffle=False)

    # Feature cols
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    # Build 2 layer fully connected DNN with 30, 30 units respectively.
    model_folder = "/tmp/hidro"
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[25, 30],
                                          model_dir=model_folder)

    # Train if we dont have a trained model
    if not os.path.isdir(model_folder) and not os.path.exists(model_folder):
        regressor.train(input_fn=get_input_fn(training_set), steps=5000)

    # Evaluate loss over one epoch of test_set.
    ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    # Print out predictions
    y = regressor.predict(input_fn=predict_func)
    # .predict() returns an iterator of dicts; convert to a list and print
    predictions = list(p["predictions"] for p in itertools.islice(y, 1))
    print("Predictions: {}".format(str(predictions[0][0])))
    return loss_score, predictions[0][0]
