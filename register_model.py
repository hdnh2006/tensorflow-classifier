#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Aug 17 15:17:13 2023

@author: henry
"""

import wandb
import random
import argparse
import logging
import sys
import os
import json
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Setup logging
script_name = Path(__file__).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{script_name}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def register_wandb_run(args):
    """Logging model metrics and saving a model to W&B."""
    
    logging.info("Starting the W&B run registration...")
    
    try:
        # Load logging model metrics
        f = open(args.metrics)
        data = json.load(f) 
    except Exception as e:
        logging.error(f"Failed to load metrics from {args.metrics}. Error: {e}")
        return

    logging.info(f"Metrics loaded from {args.metrics}.")
    
    try:
        with wandb.init(project=args.project) as run:

            # Upload logging model metrics
            run.log(data)

            # Save model to W&B
            best_model = wandb.Artifact(f'model_{run.id}', type='model')
            best_model.add_file(args.model)
            run.log_artifact(best_model)

            # Link the model to the Model Registry
            run.link_artifact(best_model, f'model-registry/{args.model_name}')

            run.finish()
            
    except Exception as e:
        logging.error(f"Failed to register W&B run. Error: {e}")
        return

    logging.info("W&B run registration completed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Simulate W&B run.')
    parser.add_argument('--project', type=str, default='bird_classifier', help='Name of the W&B project.') 
    parser.add_argument('--metrics', type=str, default='metrics.json', help='Json file with all the metrics.')
    parser.add_argument('--model', type=str, default= 'aiy_vision_classifier_birds_V1_1/saved_model.pb', help='File of weights')
    parser.add_argument('--model-name', type=str, default='aiy_vision_classifier_birds_V1_1', help='name to register')
    
    args = parser.parse_args()

    logging.info(f"Arguments received: {args}")
    register_wandb_run(args)

if __name__ == '__main__':
    main()
