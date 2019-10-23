import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import config as cf
from targets import target_distribution_gen_all
from utils_nn import np_distance, np_euclidean_distance, single_run, single_evaluation

def single_sweep_training():
    """ Goes through all target distributions in cf.pnn.target_distributions once with the config values given in config.py """

    # Loop through parameters. Convention is to start from right, since that is the least noisy distribution if I do a noise scan.
    for i in range(cf.pnn.target_distributions.shape[0]-1,-1,-1):

        # Set up new distribution
        cf.pnn.change_p_target(i)
        print('\nIn sweep {}.\nAt round {} of {} (decreasing!), with distribution {} of param {}. Target distribution:\n{}'.format(cf.pnn.sweep_id,i,cf.pnn.target_distributions.shape[0]-1,cf.pnn.target_distr_name, cf.pnn.target_ids[i],cf.pnn.p_target))

        # Run and evaluate model
        model = single_run()
        result = single_evaluation(model)

        # If we loaded weights from somewhere, then compare new distance to previous one in order to know whether new model is better than previous one.
        update_needed = True

        if cf.pnn.start_from is not None: # skips this comparison if I was in a fresh_start
            new_distance = np_distance(result, cf.pnn.p_target)
            if new_distance > cf.pnn.distances[i]:
                update_needed = False
                print("Moving on. Distance didn't improve from {}.".format(cf.pnn.distances[i]))
            else:
                print("Distance imporved! This distance:", new_distance)
        else:
            print("This distance:", np_distance(result, cf.pnn.p_target))

        if update_needed:
            # Update results
            model.save(cf.pnn.savebestpath)
            cf.pnn.distributions[i,:] = result
            cf.pnn.distances[i] = np_distance(result, cf.pnn.p_target)
            cf.pnn.euclidean_distances[i] = np_euclidean_distance(result, cf.pnn.p_target)

        # Plot graphs only if update is needed or at the end of the sweep (in case none of them were updated.)
        if update_needed or i==0:
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


if __name__ == '__main__':
    # Create directories for saving stuff
    for dir in ['saved_models', 'saved_results', 'saved_configs', 'figs_distributions', 'figs_training_sweeps']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    cf.initialize()

    # Each sweep goes through all distributions. We use different optimizer parameters in different sweeps, and load previous models.
    for sweep_id in range(20):
        cf.pnn.sweep_id = sweep_id

        # Figure out where to start from.
        if sweep_id!=0:
            cf.pnn.set_starting_points(broadness_left=cf.pnn.target_ids.shape[0], broadness_right=cf.pnn.target_ids.shape[0])

        # Change optimizer
        if sweep_id == 9:
            cf.pnn.optimizer = 'sgd'
            cf.pnn.lr = 1
            cf.pnn.decay = 0
            cf.pnn.momentum = 0.2

        if sweep_id >= 10:
            cf.pnn.lr = cf.pnn.lr * 0.4

        # Run single sweep
        single_sweep_training()
