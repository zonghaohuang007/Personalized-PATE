import comet_ml
from comet_ml import Experiment
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow import keras
import argparse
from sklearn.model_selection import train_test_split
from math import exp, sqrt, log
from model import *


def moments_accountant(eps, delta, n_step):
    x = n_step*(eps**2)/4/log(1.25/delta)
    tau = 1
    eps_ = 0
    for i in range(100):
        if tau <= 0:
            tau = 1
        eps_ = (x*(tau**2) + x*tau - log(delta))/tau
        tau = round((eps_-x)/2/x)
    return eps_


def privacy_allocation(eps, delta, n_step):
    x = eps / sqrt(n_step)
    eps_ = moments_accountant(x, delta, n_step)
    while eps_ < eps*(1-1e-6) or eps_ > eps*(1+1e-6):
        if eps_ < eps*(1-1e-6):
            x = x*1.001
            eps_ = moments_accountant(x, delta, n_step)
        else:
            x = x*0.999
            eps_ = moments_accountant(x, delta, n_step)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='knowledge_transfer_and_student_train')
    parser.add_argument('--batch_size', type=float, default=128)
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--n_query', type=int, default=1000)
    parser.add_argument('--n_histogram', type=int, default=1)
    parser.add_argument('--n_sample', type=int, default=None)
    parser.add_argument('--n_teacher', type=int, default=250)
    parser.add_argument('--p_preference', type=str, default=None)
    parser.add_argument('--delta', type=float, default=1e-5)
    args = parser.parse_args()

    exp_name = args.exp_name + '_' + args.p_preference

    # load mnist data
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    # load privacy preference
    with open('./saved_models/teacher_models_' + args.p_preference + '/privacy.pickle', 'rb') as f:
        privacy = pickle.load(f)
    assert len(privacy) == args.n_teacher

    if args.n_sample is None:
        args.n_sample = args.n_teacher

    # split the test data into two groups: unlabeled data for student model training and evaluation data
    # xtrain, xtest, ytrain_, ytest = train_test_split(x_test, y_test, train_size=args.n_query)
    n_perclass = args.n_query // args.n_class
    count = np.zeros(args.n_class)
    xtrain = []
    ytrain_ = []
    xtest = []
    ytest = []
    for i in range(len(x_test)):
        if count[y_test[i]] < n_perclass:
            xtrain.append(x_test[i])
            ytrain_.append(y_test[i])
            count[y_test[i]] = count[y_test[i]] + 1
        else:
            xtest.append(x_test[i])
            ytest.append(y_test[i])
    print(count)

    # set up comet
    print('==> setting up comet experiment...')
    experiment = Experiment(project_name='Personalized PATE', auto_param_logging=False,
                            auto_metric_logging=False,
                            parse_args=False)
    comet_ml.config.experiment = None
    experiment.set_name(exp_name)
    experiment.add_tag('')
    experiment.log_parameters(vars(args))

    # privacy allocation
    p_per_step = np.zeros(len(privacy))
    for i in range(len(privacy)):
        p_per_step[i] = privacy_allocation(
            eps=privacy[i], delta=args.delta, n_step=args.n_query*args.n_histogram*args.n_sample/args.n_teacher)

    print('============== knowledge transfer ==============')
    # aggregation
    acc = 0
    ytrain = []
    for k in range(args.n_query):
        print('==> making the {}-th query to the teacher models'.format(k))
        agg = np.zeros(args.n_class)
        # target data
        data = tf.expand_dims([xtrain[k]], axis=-1)
        data = tf.cast(data, tf.float32) / 255.
        for q in range(args.n_histogram):
            # build up model
            model = Model(args.n_class)
            model.build(input_shape=(None, 28, 28, 1))
            # model.summary()
            # sample teacher models
            sampled_teacher = np.random.choice(args.n_teacher, args.n_sample, replace=False)
            sampled_privacy = [p_per_step[i] for i in sampled_teacher]
            min_p = min(sampled_privacy)
            # histogram
            hist = np.zeros(args.n_class)
            for j in sampled_teacher:
                model.load_weights(
                    './saved_models/teacher_models_' + args.p_preference + f'/teacher_{j}')
                logits = model(data)
                pred = tf.math.argmax(logits, axis=1)[0]
                hist[int(pred.numpy())] = hist[int(pred.numpy())] + 1
            # add perturbation to the histogram
            sigma = sqrt(2*log(1.25/args.delta))/min_p
            noise = np.random.normal(loc=0.0, scale=sigma, size=args.n_class)
            hist = hist + noise
            # get the label
            label = hist.argmax()
            agg[label] = agg[label] + exp(min_p)
        # weighted aggregation
        aggregated_label = agg.argmax()
        ytrain.append(aggregated_label)
        if ytrain_[k] == aggregated_label:
            acc = acc + 1
    print('==> accuracy of the aggregated labels is {}'.format(acc/args.n_query))

    with open('./data/labeled_data_'+args.p_preference+'.pickle', 'wb') as f:
        pickle.dump([xtrain, ytrain_, xtest, ytest], f)

    # student model training
    print('====== training of student models ======')

    # build up model
    model0 = Model(args.n_class)
    model0.build(input_shape=(None, 28, 28, 1))
    model0.summary()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # prepare data
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(args.n_query)
    ds_train = ds_train.batch(args.batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((xtest, ytest))
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.shuffle(len(xtest))
    ds_test = ds_test.batch(args.batch_size)
    # ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # metrics
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    t_accuracy = tf.keras.metrics.CategoricalAccuracy()

    # train by supervised learning
    global_step = 0
    for epoch in range(args.n_epoch):
        for xtrain, ytrain in ds_train:
            xtrain = tf.expand_dims(xtrain, axis=-1)
            ytrain = tf.one_hot(ytrain, args.n_class)
            with tf.GradientTape() as tape:
                logits = model0(xtrain)
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(ytrain, logits)
                accuracy.update_state(ytrain, tf.math.softmax(logits))
            gradients = tape.gradient(loss, model0.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, model0.trainable_variables))
            global_step = global_step + 1
            if global_step % 20 == 0:
                xtest, ytest = next(iter(ds_test))
                xtest = tf.expand_dims(xtest, axis=-1)
                ytest = tf.one_hot(ytest, args.n_class)
                logits = model0(xtest)
                t_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(ytest, logits)
                t_accuracy.update_state(ytest, tf.math.softmax(logits))
                print(f'==> train student | epoch: {epoch} | step: {global_step} | loss: {loss.numpy()} | '
                      f'accuracy: {accuracy.result().numpy()} | test loss: {t_loss.numpy()} | '
                      f'test accuracy: {t_accuracy.result().numpy()}')
    model0.save_weights(f'./saved_models/student_' + args.p_preference + '/student')
    print(f'==> saved student model')
    # test student model
    test_acc = 0
    for xtest, ytest in ds_test:
        xtest = tf.expand_dims(xtest, axis=-1)
        ytest = tf.one_hot(ytest, args.n_class)
        logits = model0(xtest)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(ytest, logits)
        accuracy.update_state(ytest, tf.math.softmax(logits))
        test_acc = test_acc + accuracy.result().numpy()
    test_acc = test_acc / len(ds_test)
    print(f'==> the test accuracy of the student model is {test_acc}')

    experiment.send_notification(f'finished', 'finished')
    experiment.end()
    print('==> finished')
