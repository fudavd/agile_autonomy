#!/usr/bin/env python3

import argparse
import os

import yaml
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

sys.path.append("./src/PlannerLearning/models")

from plan_learner import PlanLearner
from config.settings import create_settings
import itertools


def main():
    parser = argparse.ArgumentParser(description='Train Planning Network')
    parser.add_argument('--settings_file',
                        help='Path to settings yaml', required=True)
    args = parser.parse_args()
    settings_filepath = args.settings_file

    batch_sizes = [8, 32, 64]
    n_epochs = [50, 100, 200]
    learning_rates = [0.03, 0.003, 0.0003]
    combine = [batch_sizes, n_epochs, learning_rates]
    parameters = list(itertools.product(*combine))
    for ind, (batch_size, epochs, learning_rate) in enumerate(parameters):
        yaml_file_dir = os.path.join('ckpt', 'sweep', str(ind))
        if not os.path.isdir(yaml_file_dir):
            os.makedirs(yaml_file_dir)
        with open(settings_filepath, 'r') as stream:
            settings = yaml.safe_load(stream)
            settings['train']['batch_size'] = batch_size
            settings['train']['max_training_epochs'] = epochs
            settings['train']['learning_rate'] = learning_rate

            settings['log_dir'] = yaml_file_dir

        run_yaml = os.path.join(yaml_file_dir, 'config.yaml')
        with open(run_yaml, 'w') as outfile:
            yaml.dump(settings, outfile, default_flow_style=False)
        run_settings = create_settings(run_yaml, mode='train')
        learner = PlanLearner(settings=run_settings)
        learner.train()


if __name__ == "__main__":
    main()
