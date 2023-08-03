#!/usr/bin/env python3

import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys

import wandb

sys.path.append("./src/PlannerLearning/models")
import time

from plan_learner import PlanLearner

from config.settings import create_settings

sweep_config = {
    'method': 'grid'
    }

metric = {
    'name': 'loss_to',
    'goal': 'minimize',
    }

parameters_dict = {
    'learning_rate': {
        'values': [0.03, 0.003, 0.0003]
        },
    'batch_size': {
          'values': [8, 16, 32, 64]
        },
    'epochs': {
          'values': [50, 100, 200]
        },
    }

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="agile-depth")

def main():
    parser = argparse.ArgumentParser(description='Train Planning Network')
    parser.add_argument('--settings_file',
                        help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='train')

    wandb.init(project='agile-depth', name='depth')
    config = wandb.config
    settings.max_training_epochs = config.epochs
    settings.batch_size = config.batch_size
    settings.learning_rate = config.learning_rate
    settings['wandb'] = wandb
    learner = PlanLearner(settings=settings)
    learner.train()


if __name__ == "__main__":
    wandb.agent(sweep_id, function=main, count=10)
