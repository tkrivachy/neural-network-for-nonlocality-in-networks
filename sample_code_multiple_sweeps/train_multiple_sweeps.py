import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

import config as cf
from targets import target_distribution_gen_all
from utils_nn import np_distance, np_euclidean_distance, single_run, single_evaluation, update_results

if __name__ == '__main__':
    # Create directories for saving stuff
    for dir in ['saved_models', 'saved_results', 'saved_configs', 'figs_distributions', 'figs_training_sweeps']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    cf.initialize()

    # Try picking up from where training was left off. If not possible, then don't load anything, just start fresh.
    try:
        cf.pnn = cf.load_config("most_recent_pnn")
        print("\nPicking up from where we left off!\n")
        starting_sweep_id = cf.pnn.sweep_id + 1
    except FileNotFoundError:
        print("\nStarting fresh!\n")
        starting_sweep_id = cf.pnn.sweep_id

    # Each sweep goes through all distributions. We use different optimizer parameters in different sweeps, and load previous models.
    for sweep_id in range(starting_sweep_id, 12):
        cf.pnn.sweep_id = sweep_id

        # Set parameters of this training sweep.

        ## For a few sweeps, reinitialize completely.
        if cf.pnn.sweep_id<=1:
            cf.pnn.set_starting_points(fresh_start=True)

        ## Then for a few sweeps, start from previous best model for that distribution.
        if cf.pnn.sweep_id> 1:
            cf.pnn.set_starting_points(broadness_left=0, broadness_right=0)

        ## After a given number of sweeps, learn from all other models.
        if cf.pnn.sweep_id >= 3:
            cf.pnn.set_starting_points(broadness_left=cf.pnn.target_ids.shape[0], broadness_right=cf.pnn.target_ids.shape[0])

        ## Change to SGD.
        if cf.pnn.sweep_id == 7:
            cf.pnn.optimizer = 'sgd'
            cf.pnn.lr = 1
            cf.pnn.decay = 0
            cf.pnn.momentum = 0.2

        ## Gradually reduce learning rate for SGD for fine-tuning.
        if cf.pnn.sweep_id > 7:
            cf.pnn.lr = cf.pnn.lr * 0.4

        ## Add more phases here if you'd like!
        ## E.g. increase batch size, change the loss function to a more fine-tuned one, or change the optimizer!

        # Run single sweep
        # Loop through parameters. Convention is to start from right, since that is the least noisy distribution if I do a noise scan.
        for i in range(cf.pnn.target_distributions.shape[0]-1,-1,-1):
            # Set up new distribution
            cf.pnn.change_p_target(i)
            print('\nIn sweep {}.\nAt round {} of {} (decreasing!), with distribution {} of param {}. Target distribution:\n{}'.format(cf.pnn.sweep_id,i,cf.pnn.target_distributions.shape[0]-1,cf.pnn.target_distr_name, cf.pnn.target_ids[i],cf.pnn.p_target))

            # Run model
            model = single_run()

            # If we loaded weights from somewhere, then compare new distance to previous one in order to know whether new model is better than previous one.
            update_results(model,i)
        # Save config of the most recently finished sweep. We will continue from here if train_multiple_sweeps is run again.
        cf.pnn.save("sweep_"+str(cf.pnn.sweep_id)+"_pnn")
        cf.pnn.save("most_recent_pnn")
