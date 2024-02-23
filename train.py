import sys
import argparse
from model import get_model
import tensorflow as tf
import random
import cv2
import numpy as np
from trainer import prepare_dataset, train_model

parser = argparse.ArgumentParser(description="Training parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-m", "--model_path", help="path to model dir", default='./final_model/', type=str)
parser.add_argument("-c", "--model_config", help="path to model config", type=str)

args = vars(parser.parse_args())

model_path = args['model_path']

model_config = {'name':'model...', 'checkpoint_dir' : './models/ndr/', 'use_pretrained':False,'n_epochs':5,
                'batch_size':4, 'img_size' : (768,768), 'num_classes': 2} if args['model_config'] is None else eval(args['model_config'])
image_path = args['image_path']


model = get_model(model_config['img_size'], model_config['num_classes'])
train_dataset, valid_dataset = prepare_dataset()
checkpoint = train_model(model, **model_config)

print(f'Your model is saved at {checkpoint}')