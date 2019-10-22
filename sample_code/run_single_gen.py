import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from scipy.stats import entropy

import config as cf
from utils_nn import customLoss, generate_xy_batch, generate_x_test, customLoss_distr, build_model
from targets import target_distribution_gen

def single_evaluation(model):
    """ Evaluates the model and returns the resulting distribution as a numpy array. """
    test_pred = model.predict_generator(generate_x_test(), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    result = K.eval(customLoss_distr(test_pred))
    return result

def single_run():
    """ Runs training algorithm for a single target distribution. Returns model."""
    # Stuff that needs to be removed:
    cf.pnn.LR_reduction_patience = 5
    cf.pnn.LR_reduction_factor = 0.99
    cf.pnn.plot_before_epochs = False

    # Model and optimizer related setup.
    K.clear_session()
    model = build_model()
    if cf.pnn.start_from is not None:
        print("LOADING MODEL WEIGHTS FROM", cf.pnn.start_from)
        model = load_model(cf.pnn.start_from,custom_objects={'customLoss': customLoss})

    if cf.pnn.optimizer.lower() == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(lr=cf.pnn.lr, rho=0.95, epsilon=None, decay=cf.pnn.decay)
    elif cf.pnn.optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.SGD(lr=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
        print("\n\nWARNING!!! Optimizer {} not recognized. Please implement it if you want to use it. Using SGD instead.\n\n".format(cf.pnn.optimizer))
        cf.pnn.optimizer = 'sgd' # set it for consistency.

    model.compile(loss=customLoss, optimizer = optimizer, metrics=[])

    # Fit and save model
    model.fit_generator(generate_xy_batch(), steps_per_epoch=cf.pnn.no_of_batches, epochs=1, verbose=1, validation_data=generate_xy_batch(), validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    model.save(cf.pnn.savebestpath)
    return model
