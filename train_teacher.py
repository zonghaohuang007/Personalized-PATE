import comet_ml
from comet_ml import Experiment
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow import keras
import argparse
from model import *
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='train_teacher')
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--n_teacher', type=int, default=250)
    parser.add_argument('--p_preference', type=str, default=None)
    args = parser.parse_args()

    exp_name = args.exp_name + '_' + args.p_preference

    # load mnist data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    # set up comet
    print('==> setting up comet experiment...')
    experiment = Experiment(project_name='Personalized PATE', auto_param_logging=False,
                            auto_metric_logging=False,
                            parse_args=False)
    comet_ml.config.experiment = None
    experiment.set_name(exp_name)
    experiment.add_tag('')
    experiment.log_parameters(vars(args))

    # load personalized privacy preferecne
    privacy_path = './data/' + args.p_preference + '.pickle'
    with open(privacy_path, 'rb') as f:
        privacy = pickle.load(f)
    assert len(privacy) == 60000, 'the length of privacy preference does not match the length of data'
    if args.p_preference != 'p5':
        sorted_index = np.argsort(privacy)
    else:
        sorted_index = list(range(60000))
        random.shuffle(sorted_index)
    # sorted_index = np.random.choice(60000, 60000, replace=False)

    # split dataset into m groups
    split_x = []
    split_y = []
    split_p = []
    n_per_teacher = len(x_train) // args.n_teacher
    count = np.zeros(args.n_teacher)
    # count_ = np.zeros((args.n_teacher, args.n_class))
    for i in range(args.n_teacher):
        split_x.append([])
        split_y.append([])
        split_p.append([])
    k = 0
    for i in range(len(x_train)):
        j = y_train[sorted_index[i]]  # label of the data
        split_x[k].append(x_train[sorted_index[i]])
        split_y[k].append(y_train[sorted_index[i]])
        split_p[k].append(privacy[sorted_index[i]])
        # count_[k][j] = count_[k][j] + 1
        count[k] = count[k] + 1
        if count[k] == n_per_teacher:
            k = k + 1  # index of the teacher/group
    p_teacher = np.zeros(args.n_teacher)
    for i in range(args.n_teacher):
        p_teacher[i] = min(split_p[i])
    print('==> privacy budgets of teacher models:')
    print(p_teacher)

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.shuffle(len(x_test))
    ds_test = ds_test.batch(args.batch_size)
    # ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # teacher models training
    average_acc = 0
    for i in range(args.n_teacher):
        print(f'====== training of {i+1}-th teacher models ======')
        ds_train = tf.data.Dataset.from_tensor_slices((split_x[i], split_y[i]))
        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        # ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(n_per_teacher)
        ds_train = ds_train.batch(args.batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        # build up model
        model = Model(args.n_class)
        model.build(input_shape=(None, 28, 28, 1))
        # model.summary()
        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        # metrics
        accuracy = tf.keras.metrics.CategoricalAccuracy()
        t_accuracy = tf.keras.metrics.CategoricalAccuracy()
        # train
        global_step = 0
        for epoch in range(args.n_epoch):
            for xtrain, ytrain in ds_train:
                # xtrain, ytrain = next(iter(ds_train))
                xtrain = tf.expand_dims(xtrain, axis=-1)
                ytrain = tf.one_hot(ytrain, args.n_class)
                with tf.GradientTape() as tape:
                    logits = model(xtrain)
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(ytrain, logits)
                    accuracy.update_state(ytrain, tf.math.softmax(logits))
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
                global_step = global_step + 1
                if global_step % 20 == 0:
                    xtest, ytest = next(iter(ds_test))
                    xtest = tf.expand_dims(xtest, axis=-1)
                    ytest = tf.one_hot(ytest, args.n_class)
                    logits = model(xtest)
                    t_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(ytest, logits)
                    t_accuracy.update_state(ytest, tf.math.softmax(logits))
                    print(f'==> train {i+1}-th teacher | epoch: {epoch} | step: {global_step} | loss: {loss.numpy()} | '
                          f'accuracy: {accuracy.result().numpy()} | test loss: {t_loss.numpy()} | '
                          f'test accuracy: {t_accuracy.result().numpy()}')
        average_acc = average_acc + t_accuracy.result().numpy()
        model.save_weights('./saved_models/teacher_models_' + args.p_preference + f'/teacher_{i}')
        print(f'==> saved {i+1}-th teacher model')
    with open('./saved_models/teacher_models_' + args.p_preference + '/privacy.pickle', 'wb') as f:
        pickle.dump(p_teacher, f)
    print('==> averaged test accuracy is {}'.format(average_acc/args.n_teacher))
    experiment.send_notification(f'finished', 'finished')
    experiment.end()
    print('==> finished')
