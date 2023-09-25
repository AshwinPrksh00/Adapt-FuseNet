import argparse
import os
import tensorflow as tf
from tensorflow.keras import *
from dataLoader import get_face_data
from model import build_face_model
from keras.callbacks import ModelCheckpoint


# Parse arguments for number of epochs, number of classes, batch size, saved model path, input shape to the model, path to model weights, path to dataset
parser = argparse.ArgumentParser(description='Train Face Model')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs, default=100')
parser.add_argument('--num_classes', type=int, default=50, help='Number of classes, default=50')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size, default=4')
parser.add_argument('--saved_model_path', type=str, default='Model', help='Path to save the model, default = Model')
parser.add_argument('--input_shape', type=tuple, default=(128, 128, 3), help='Input shape to the model, default=(128, 128, 3)')
parser.add_argument('--weight_model', type=str, default=None, help='Path to the model weights, default=None')
parser.add_argument('--dataset_path', type=str, default='Data', help='Path to the dataset, default=Data')

args = parser.parse_args()

# set the number of epochs, number of classes, batch size, saved model path

n_epochs = args.epochs
num_classes = args.num_classes
batch_size = args.batch_size
saved_model_path = args.saved_model_path
input_shape = args.input_shape
dataset_path = args.dataset_path

#Check if the saved model path exists, if not create one
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

#Load the dataset
x_train, y = get_face_data(num_classes, face_angles=['90_1',  '90_2', '90_3', '90_4'], path=dataset_path)
x_val, y_val = get_face_data(num_classes, face_angles=['90_5'], path=dataset_path)
x_test, y_test = get_face_data(num_classes, face_angles=['90_6'], path=dataset_path)

# build the model
model = build_face_model(input_shape, num_classes)

# load the weights if provided
if args.weight_model:
    model.load_weights(args.weight_model)

# learning rate scheduler
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 400*((len(x_train)*0.8)/4), 1e-5)
lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
es = tf.keras.callbacks.EarlyStopping(
monitor="val_loss",
min_delta=0,
patience=40,
verbose=0,
mode="auto",
baseline=None,
restore_best_weights=False,
start_from_epoch=200,
)

#Chekpointing the model
checkpoint = ModelCheckpoint(filepath=f'/{saved_model_path}/face_model_feature_trial.h5', 
                    monitor='val_loss',
                    verbose=1, 
                    save_best_only=True,
                    mode='min')

# Fit the model on the training data.
model.fit(x_train, y,
        batch_size = batch_size, 
        epochs = n_epochs,
        callbacks=[checkpoint, es], 
        validation_data = (x_val, y_val), 
        validation_batch_size = batch_size,
        verbose=1)

# Evaluate the model accuracy on the validation set.
score = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy of Model:', score[1])