import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import config as cf
from targets import target_distribution_gen_all
from utils_nn import np_distance, np_euclidean_distance, single_run, single_evaluation, plot_strategies

def single_sweep_training():
    """ Goes through all target distributions in cf.pnn.target_distributions once with the config values given in config.py """

    # Loop through parameters. Convention is to start from right, since that is the least noisy distribution if I do a noise scan.
    for i in range(cf.pnn.target_distributions.shape[0]-1,-1,-1):

        # Set up new distribution
        cf.pnn.change_p_target(i)
        print('\n\n At round {} of {} (decreasing!), with distribution {} of param {}. Target distribution:\n{}'.format(i+1,cf.pnn.target_distributions.shape[0],cf.pnn.target_distr_name, cf.pnn.target_ids[i],cf.pnn.p_target))

        # Run and evaluate model
        model = single_run()
        result = single_evaluation(model)

        # Update results
        model.save(cf.pnn.savebestpath)
        cf.pnn.distributions[i,:] = result
        cf.pnn.distances[i] = np_distance(result, cf.pnn.p_target)
        cf.pnn.euclidean_distances[i] = np_euclidean_distance(result, cf.pnn.p_target)
        print("This distance:", cf.pnn.euclidean_distances[i])

        # Plot distances
        plt.clf()
        fig = plt.figure(figsize=(6.4, 4.8))
        plt.plot(cf.pnn.target_ids,cf.pnn.euclidean_distances, 'ro')
        plt.title("D(p_target,p_machine)")
        if i!=0:
            plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.euclidean_distances))[-2]*1.2)
        else:
            plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.euclidean_distances))[-1]*1.2)
        plt.savefig("./figs_training_sweeps/sweep.png")

        # Plot distributions
        plt.clf()
        plt.style.use('default')
        plt.plot(cf.pnn.p_target,'ro',markersize=5)
        plt.plot(result,'gs',alpha = 0.85,markersize=5)
        plt.title("Target distr.: {} {:.3f}".format(cf.pnn.target_distr_name, cf.pnn.target_ids[i]))
        plt.ylim(bottom=0,top=max(cf.pnn.p_target)*1.2)
        plt.savefig("./figs_distributions/target_"+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_distributions.shape[0]))))+".png")

        # Plots the strategies (comment out if you're not particularly interested - it's a bit slow.)
        plot_strategies(i)

if __name__ == '__main__':
    # Create directories for saving stuff
    for dir in ['saved_models', 'saved_results', 'saved_configs', 'figs_distributions', 'figs_training_sweeps','figs_strategies']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    cf.initialize()
    # Run single sweep
    single_sweep_training()
