import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy

import config as cf
from targets import target_distribution_gen

def build_model():
    """ Build NN """
    inputTensor = Input((3,))
    group_lambda = Lambda(lambda x: x[:,:1], output_shape=((1,)))(inputTensor)
    group_x_hidden = Lambda(lambda x: x[:,1:2], output_shape=((1,)))(inputTensor) # a input
    group_y_hidden = Lambda(lambda x: x[:,2:3], output_shape=((1,)))(inputTensor) # c input
    #group_x = K.squeeze(K.one_hot(K.cast(group_x_hidden,'int32'), pnn.ainputsize),axis=1)
    #group_z = K.squeeze(K.one_hot(K.cast(group_z_hidden,'int32'), pnn.cinputsize),axis=1)
    ais = cf.pnn.ainputsize
    bis = cf.pnn.binputsize
    group_x_hidden = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), ais),axis=1) , output_shape=((ais,)))(group_x_hidden)
    group_y_hidden = Lambda(lambda x:K.squeeze(K.one_hot(K.cast(x,'int32'), bis),axis=1) , output_shape=((bis,)))(group_y_hidden)
    #group_x_hidden = K.squeeze(K.one_hot(K.cast(group_x_hidden,'int32'), pnn.ainputsize),axis=1)
    #group_z_hidden = K.squeeze(K.one_hot(K.cast(group_z_hidden,'int32'), pnn.cinputsize),axis=1)
    group_x = group_x_hidden
    group_y = group_y_hidden

    amean = ais/2
    astd = np.sqrt((ais**2-1)/12)
    bmean = bis/2
    bstd = np.sqrt((bis**2-1)/12)

    group_x_hidden = Lambda(lambda x: (x-amean)/astd , output_shape=((ais,)))(group_x_hidden)
    group_y_hidden = Lambda(lambda x: (x-bmean)/bstd , output_shape=((bis,)))(group_y_hidden)
    #group_x_hidden = (group_x_hidden - 0.5)*2
    #group_z_hidden = (group_z_hidden - 0.5)*2

    for _ in range(cf.pnn.greek_depth):
        group_lambda = Dense(cf.pnn.greek_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg)(group_lambda)

    #group_x_hidden = (group_x_hidden - 1.5)/1.11803398875
    #group_z_hidden = (group_z_hidden - 1.5)/1.11803398875
    group_a = Concatenate()([group_lambda,group_x_hidden])
    group_b = Concatenate()([group_lambda,group_y_hidden])

    ## Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    #kernel_init = tf.keras.initializers.VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)
    """"""""""""""""""""""""""""""
    from tensorflow.keras.initializers import VarianceScaling
    kernel_init = VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)
    """"""""""""""""""""""""""""""
    for _ in range(cf.pnn.latin_depth):
        group_a = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_a)
        group_b = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_b)

    group_a = Dense(cf.pnn.a_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_a)
    group_b = Dense(cf.pnn.b_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_b)

    #outputTensor = Concatenate()([group_a,group_b,group_c,a_input,c_input])
    outputTensor = Concatenate()([group_x,group_y,group_a,group_b])

    model = Model(inputTensor,outputTensor)
    return model

def np_euclidean_distance(p,q=0):
    """ Euclidean distance, useful for plotting results."""
    return np.sqrt(np.sum(np.square(p-q),axis=-1))

def np_distance(p,q=0):
    """ Same as the distance used in the loss function, just written for numpy arrays.
    Implemented losses:
        l2: Euclidean distance (~Mean Squared Error)
        l1: L1 distance (~Mean Absolute Error)
        kl: Kullback-Liebler divergence (relative entropy)
        js: Jensen-Shannon divergence (see https://arxiv.org/abs/1803.08823 pg. 94-95). Thanks to Askery A. Canabarro for the recommendation.
    """
    if cf.pnn.loss.lower() == 'l2':
        return np.sum(np.square(p-q),axis=-1)
    elif cf.pnn.loss.lower() == 'l1':
        return 0.5*np.sum(np.abs(p-q),axis=-1)
    elif cf.pnn.loss.lower() == 'kl':
        p = np.clip(p, K.epsilon(), 1)
        q = np.clip(q, K.epsilon(), 1)
        return np.sum(p * np.log(np.divide(p,q)), axis=-1)
    elif cf.pnn.loss.lower() == 'js':
        p = np.clip(p, K.epsilon(), 1)
        q = np.clip(q, K.epsilon(), 1)
        avg = (p+q)/2
        return np.sum(p * np.log(np.divide(p,avg)), axis=-1) + np.sum(q * np.log(np.divide(q,avg)), axis=-1)

def keras_distance(p,q):
    """ Distance used in loss function.
    Implemented losses:
        l2: Euclidean distance (~Mean Squared Error)
        l1: L1 distance (~Mean Absolute Error)
        kl: Kullback-Liebler divergence (relative entropy)
        js: Jensen-Shannon divergence (see https://arxiv.org/abs/1803.08823 pg. 94-95). Thanks to Askery A. Canabarro for the recommendation.
    """
    if cf.pnn.loss.lower() == 'l2':
        return K.sum(K.square(p-q),axis=-1)
    elif cf.pnn.loss.lower() == 'l1':
        return 0.5*K.sum(K.abs(p-q), axis=-1)
    elif cf.pnn.loss.lower() == 'kl':
        p = K.clip(p, K.epsilon(), 1)
        q = K.clip(q, K.epsilon(), 1)
        return K.sum(p * K.log(p / q), axis=-1)
    elif cf.pnn.loss.lower() == 'js':
        p = K.clip(p, K.epsilon(), 1)
        q = K.clip(q, K.epsilon(), 1)
        avg = (p+q)/2
        return K.sum(p * K.log(p / avg), axis=-1) + K.sum(q * K.log(q / avg), axis=-1)

def customLoss_distr(y_pred):
    x_probs = y_pred[:,0:cf.pnn.ainputsize]
    y_probs = y_pred[:,cf.pnn.ainputsize:cf.pnn.ainputsize+cf.pnn.binputsize]
    temp_start = cf.pnn.ainputsize+cf.pnn.binputsize
    a_probs = y_pred[:,temp_start:temp_start + cf.pnn.a_outputsize]
    b_probs = y_pred[:,temp_start +cf.pnn.a_outputsize:temp_start +cf.pnn.a_outputsize + cf.pnn.b_outputsize]

    x_probs = K.reshape(x_probs,(-1,cf.pnn.ainputsize,1,1,1))
    y_probs = K.reshape(y_probs,(-1,1,cf.pnn.binputsize,1,1))
    a_probs = K.reshape(a_probs,(-1,1,1,cf.pnn.a_outputsize,1))
    b_probs = K.reshape(b_probs,(-1,1,1,1,cf.pnn.b_outputsize))

    probs = x_probs*y_probs*a_probs*b_probs
    probs = K.mean(probs,axis=0)
    probs = K.flatten(probs)
    return probs


def customLoss(y_true,y_pred):
    """ Custom loss function."""
    # Note that y_true is just batch_size copies of the target distributions. So any row could be taken here. We just take 0-th row.
    return keras_distance(y_true[0,:], customLoss_distr(y_pred))

# Set up generator for X and Y data
training_mean = 0.5
training_sigma = 0.28867513459 #= np.sqrt(1/12)

def generate_xy_batch():
    while True:
        temp = (np.random.random((cf.pnn.batch_size, 1)) - training_mean)/training_sigma
        ainputs = np.random.choice(cf.pnn.ainputsize, (cf.pnn.batch_size,1))
        binputs = np.random.choice(cf.pnn.binputsize, (cf.pnn.batch_size,1))
        temp = np.concatenate((temp, ainputs, binputs), axis=1)
        yield (temp, cf.pnn.y_true)

def generate_x_test():
    while True:
        temp = (np.random.random((cf.pnn.batch_size,1)) - training_mean)/training_sigma
        ainputs = np.random.choice(cf.pnn.ainputsize, (cf.pnn.batch_size,1))
        binputs = np.random.choice(cf.pnn.binputsize, (cf.pnn.batch_size,1))
        temp = np.concatenate((temp, ainputs, binputs), axis=1)
        yield temp

def single_evaluation(model):
    """ Evaluates the model and returns the resulting distribution as a numpy array. """
    test_pred = model.predict_generator(generate_x_test(), steps=1, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    result = K.eval(customLoss_distr(test_pred))
    return result

def single_run():
    """ Runs training algorithm for a single target distribution. Returns model."""
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

    # Fit model
    model.fit_generator(generate_xy_batch(), steps_per_epoch=cf.pnn.no_of_batches, epochs=1, verbose=1, validation_data=generate_xy_batch(), validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    return model

def compare_models(model1,model2):
    """ Evaluates two models for p_target distribution and return one which is closer to it."""
    result1 = single_evaluation(model1)
    result2 = single_evaluation(model2)
    if np_distance(result1, cf.pnn.p_target) < np_distance(result2, cf.pnn.p_target):
        return model1, 1
    else:
        return model2, 2

def update_results(model_new,i):
    """ Updates plots and results if better than the one I loaded the model from in this round.
    If I am in last sample of the sweep I will plot no matter one, so that there is at least one plot per sweep.
    """
    result_new = single_evaluation(model_new)
    distance_new = np_distance(result_new, cf.pnn.p_target)

    # Decide whether to use new or old model.
    if cf.pnn.start_from is not None: # skips this comparison if I was in a fresh_start
        try:
            model_old = load_model('./saved_models/best_'+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_distributions.shape[0]))))+'.hdf5', custom_objects={'customLoss': customLoss})
            result_old = single_evaluation(model_old)
            distance_old = np_distance(result_old, cf.pnn.p_target)
            if distance_new > distance_old:
                print("Moving on. With old model distance is at {}.".format(distance_old))
                result = result_old
                model = model_old
                distance = distance_old
            else:
                print("Distance imporved! Distance with new model:", distance_new)
                result = result_new
                model = model_new
                distance = distance_new
        except FileNotFoundError:
            print("This distance:", distance_new)
            result = result_new
            model = model_new
            distance = distance_new
    else:
        print("This distance:", distance_new)
        result = result_new
        model = model_new
        distance = distance_new

    # Update results
    model.save(cf.pnn.savebestpath)
    cf.pnn.distributions[i,:] = result
    cf.pnn.distances[i] = distance
    cf.pnn.euclidean_distances[i] = np_euclidean_distance(result, cf.pnn.p_target)
    np.save("./saved_results/target_distributions.npy",cf.pnn.target_distributions)
    np.save("./saved_results/distributions.npy",cf.pnn.distributions)
    np.save("./saved_results/distances.npy",cf.pnn.distances)
    np.save("./saved_results/euclidean_distances.npy",cf.pnn.euclidean_distances)

    # Plot distances
    plt.clf()
    plt.title("D(p_target,p_machine)")
    plt.plot(cf.pnn.target_ids,cf.pnn.euclidean_distances, 'ro')
    if i!=0 and cf.pnn.sweep_id==0:
        plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.euclidean_distances))[-2]*1.2)
    else:
        plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.euclidean_distances))[-1]*1.2)
    plt.savefig("./figs_training_sweeps/sweep"+str(cf.pnn.sweep_id)+".png")

    # Plot distributions
    plt.clf()
    plt.plot(cf.pnn.p_target,'ro',markersize=5)
    plt.plot(result,'gs',alpha = 0.85,markersize=5)
    plt.title("Target distr. (in red): {} {:.3f}".format(cf.pnn.target_distr_name, cf.pnn.target_ids[i]))
    plt.ylim(bottom=0,top=max(cf.pnn.p_target)*1.2)
    plt.savefig("./figs_distributions/target_"+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_ids.shape[0]))))+".png")
