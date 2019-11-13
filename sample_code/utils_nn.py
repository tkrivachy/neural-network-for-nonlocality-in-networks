import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda
from scipy.stats import entropy
from matplotlib.lines import Line2D

import config as cf
from targets import target_distribution_gen

def build_model():
    cf.pnn.inputsize = 3 # Number of hidden variables, i.e. alpha, beta, gamma
    """ Build NN for triangle """
    # Hidden variables as inputs.
    inputTensor = Input((cf.pnn.inputsize,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:,:1], output_shape=((1,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:,1:2], output_shape=((1,)))(inputTensor)
    group_gamma = Lambda(lambda x: x[:,2:3], output_shape=((1,)))(inputTensor)

    # Neural network at the sources, for pre-processing (e.g. for going from uniform distribution to non-uniform one)
    ## Note that in the example code greek_depth is set to 0, so this part is trivial.
    for _ in range(cf.pnn.greek_depth):
        group_alpha = Dense(cf.pnn.greek_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg)(group_alpha)
        group_beta = Dense(cf.pnn.greek_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg)(group_beta)
        group_gamma = Dense(cf.pnn.greek_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg)(group_gamma)

    # Route hidden variables to visibile parties Alice, Bob and Charlie
    group_a = Concatenate()([group_beta,group_gamma])
    group_b = Concatenate()([group_gamma,group_alpha])
    group_c = Concatenate()([group_alpha,group_beta])

    # Neural network at the parties Alice, Bob and Charlie.
    ## Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(cf.pnn.latin_depth):
        group_a = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_a)
        group_b = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_b)
        group_c = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_c)

    # Apply final softmax layer
    group_a = Dense(cf.pnn.a_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_a)
    group_b = Dense(cf.pnn.b_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_b)
    group_c = Dense(cf.pnn.c_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_c)

    outputTensor = Concatenate()([group_a,group_b,group_c])

    model = Model(inputTensor,outputTensor)
    return model

def np_euclidean_distance(p,q=0):
    """ Euclidean distance, useful for plotting results."""
    return np.sqrt(np.sum(np.square(p-q),axis=-1))

def np_distance(p,q=0):
    """ Same as the distance used in the loss function, just written for numpy arrays."""
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
    """ Distance used in loss function. """
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
    """ Converts the output of the neural network to a probability vector.
    That is from a shape of (batch_size, a_outputsize + b_outputsize + c_outputsize) to a shape of (a_outputsize * b_outputsize * c_outputsize,)
    """
    a_probs = y_pred[:,0:cf.pnn.a_outputsize]
    b_probs = y_pred[:,cf.pnn.a_outputsize : cf.pnn.a_outputsize + cf.pnn.b_outputsize]
    c_probs = y_pred[:,cf.pnn.a_outputsize + cf.pnn.b_outputsize : cf.pnn.a_outputsize + cf.pnn.b_outputsize + cf.pnn.c_outputsize]

    a_probs = K.reshape(a_probs,(-1,cf.pnn.a_outputsize,1,1))
    b_probs = K.reshape(b_probs,(-1,1,cf.pnn.b_outputsize,1))
    c_probs = K.reshape(c_probs,(-1,1,1,cf.pnn.c_outputsize))

    probs = a_probs*b_probs*c_probs
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
        temp = np.divide((np.random.random((cf.pnn.batch_size, cf.pnn.inputsize)) - training_mean),training_sigma)
        yield (temp, cf.pnn.y_true)

def generate_x_test():
    while True:
        temp = np.divide((np.random.random((cf.pnn.batch_size_test, cf.pnn.inputsize)) - training_mean),training_sigma)
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

    # Plot strategies (only turn on if you're really interested, since it takes quite a bit of time to update in each step!)
    plot_strategies(i)

def plot_strategies(i):
    sample_size = 4000 #how many hidden variable triples to sample from
    random_sample_size = 5 #for each hidden variable triple, how many times to sample from strategies.
    alpha_value = 0.25# 3/random_sample_size #opacity of dots. 0.1 or 0.25 make for nice paintings.
    markersize = 5000/np.sqrt(sample_size)

    modelpath = './saved_models/best_'+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_distributions.shape[0]))))+'.hdf5'

    input_data = generate_x_test()
    inputs = next(input_data)
    while inputs.shape[0] < sample_size:
        inputs = np.concatenate((inputs, next(input_data)),axis=0)
    inputs = inputs[:sample_size,:]

    K.clear_session()
    model = load_model(modelpath,custom_objects={'customLoss': customLoss})
    y = model.predict(inputs)

    y_a = y[:,0:cf.pnn.a_outputsize]
    y_b = y[:,cf.pnn.a_outputsize:cf.pnn.a_outputsize+cf.pnn.b_outputsize]
    y_c = y[:,cf.pnn.a_outputsize+cf.pnn.b_outputsize:cf.pnn.a_outputsize+cf.pnn.b_outputsize+cf.pnn.c_outputsize]

    y_a = np.array([np.random.choice(np.arange(cf.pnn.a_outputsize),p=y_a[j,:], size = random_sample_size) for j in range(y_a.shape[0])]).reshape(random_sample_size*sample_size)
    y_b = np.array([np.random.choice(np.arange(cf.pnn.b_outputsize),p=y_b[j,:], size = random_sample_size) for j in range(y_b.shape[0])]).reshape(random_sample_size*sample_size)
    y_c = np.array([np.random.choice(np.arange(cf.pnn.c_outputsize),p=y_c[j,:], size = random_sample_size) for j in range(y_c.shape[0])]).reshape(random_sample_size*sample_size)

    training_mean = 0.5
    training_sigma = np.sqrt(1/12)
    inputs = inputs* training_sigma + training_mean
    # Tile and reshape since we sampled random_sample_size times from each input.
    inputs = np.array(np.array([np.tile(inputs[i,:],(random_sample_size,1)) for i in range(inputs.shape[0])])).reshape(random_sample_size*sample_size,3)

    alphas = inputs[:,0]
    betas = inputs[:,1]
    gammas = inputs[:,2]
    inputs_a = np.stack((betas,gammas)).transpose()
    inputs_b = np.stack((alphas,gammas)).transpose()
    inputs_c = np.stack((alphas,betas)).transpose()

    colordict = {0:'red',1:'green',2:'blue',3:'orange'}
    colors_alice = [colordict[i] for i in y_a]
    colors_bob = [colordict[i] for i in y_b]
    colors_charlie = [colordict[i] for i in y_c]

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='0',
                              markerfacecolor='red', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='1',
                                markerfacecolor='green', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='2',
                                markerfacecolor='blue', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='3',
                                markerfacecolor='orange', markersize=8)]

    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    plt.subplot(2,2,1)
    plt.scatter(inputs_a[:,0],inputs_a[:,1], color = colors_alice, alpha=alpha_value, s = markersize)
    plt.gca().invert_yaxis()
    plt.title('Response of Alice to her inputs.')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\gamma$')

    plt.subplot(2,2,2)
    plt.scatter(inputs_b[:,0],inputs_b[:,1], color = colors_bob, alpha=alpha_value, s = markersize)
    plt.gca().invert_yaxis()
    plt.title('Response of Bob to his inputs.')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\gamma$')

    plt.subplot(2,2,3)
    plt.scatter(inputs_c[:,1],inputs_c[:,0], color = colors_charlie, alpha=alpha_value, s = markersize)
    plt.gca().invert_yaxis()
    plt.title('Response of Charlie to his inputs.')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\alpha$')

    plt.subplot(2,2,4)
    plt.plot(cf.pnn.target_distributions[i,:],'ro',markersize=5)
    plt.plot(cf.pnn.distributions[i,:],'gs',alpha = 0.85,markersize=5)
    plt.title('Target (red) and learned (green) distributions')
    plt.xlabel('outcome')
    plt.ylabel('probability of outcome')

    fig.suptitle(cf.pnn.target_distr_name +', distribution no. '+str(i), fontsize = 14)
    #fig.legend(handles=legend_elements, loc='lower right',bbox_to_anchor = (0.75,0.25))
    fig.legend(handles=legend_elements, loc='upper right')
    plt.savefig('./figs_strategies/strat_'+str(i))
