import sys
import argparse
from model import get_model
import tensorflow as tf
import random
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Evaluation parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-m", "--model_path", help="path to model dir", default='./final_model/', type=str)
parser.add_argument("-c", "--model_config", help="path to model config", type=str)
parser.add_argument("-i", "--image_path", help="path to image")
parser.add_argument("-d", "--inference_destination", help="inference destination dir")

args = vars(parser.parse_args())

model_path = args['model_path']
model_config = {'img_size' : (768,768), 'num_classes': 2} if args['model_config'] is None else args['model_config']
image_path = args['image_path']
inference_destination = args['inference_destination']


model = get_model(model_config['img_size'], model_config['num_classes'])
latest_simple_prob = tf.train.latest_checkpoint(model_path)
model.load_weights(latest_simple_prob)
random = False
if image_path is None:
    print("You have entered no path to image.\n You can choose:\n\tType 1 to inference on random photo"
          "\n\tType 2 to provide path to image")
    desicion = input()
    random = False if desicion == '2' else True
    if random:
        with open('imgs_to_inference.txt', 'r', encoding='utf-8') as f:
            a = f.read()
            files = a.split(' ')[:-1:][-1]
        file = random.choice(files)
        x_test = cv2.imread(file)
        x_test = tf.convert_to_tensor(x_test[np.newaxis, :] / 255., dtype=tf.float32)
        pred_test = model(x_test[:])
        tens = pred_test.numpy()
        cv2.imwrite(inference_destination if inference_destination is not None else 'output.jpg',
                    tens)
    else:
        file = input('provide filename')
        x_test = cv2.imread(file)
        x_test = tf.convert_to_tensor(x_test[np.newaxis, :] / 255., dtype=tf.float32)
        pred_test = model(x_test[:])
        tens = pred_test.numpy()
        cv2.imwrite(inference_destination if inference_destination is not None else 'output.jpg',
                    tens)