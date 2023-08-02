#!/usr/bin/env python3

import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append("./src/PlannerLearning/models")
import time

from plan_learner import PlanLearner

from config.settings import create_settings


def main():
    parser = argparse.ArgumentParser(description='Train Planning Network')
    parser.add_argument('--settings_file',
                        help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='train')

    learner = PlanLearner(settings=settings)
    learner.train()


if __name__ == "__main__":
    main()
