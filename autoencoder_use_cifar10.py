"""
MIT License

Copyright (c) 2016 deeplearningathome. http://deeplearningathome.com/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tensorflow as tf
from autoencoder import AutoEncoder
import numpy
from datetime import datetime
import time

flags = tf.flags
flags.DEFINE_string("encoder_network", "3072,1024,128", "specifies encoder network")
flags.DEFINE_float("noise_level", 0.0, "noise level for denoising autoencoder")
flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 1000000, "Number of batches to run.")
flags.DEFINE_string("acitivation_kind", "sigmoid", "type of neuron activations")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_string("optimizer_kind", "adamoptimizer", "type of oprtimizer")
flags.DEFINE_string("logdir", "tblogs", "tensorboard logs")
flags.DEFINE_string("data_dir", "cifar10data", "location of cifar10 data")
FLAGS = flags.FLAGS

from utils_cifar10 import maybe_download_and_extract, inputs, distorted_inputs, IMAGE_SIZE, NUM_CLASSES
from utils import variable_summaries
PIXEL_DEPTH = 255

#using convolutions
def conv_main(_):
    maybe_download_and_extract(FLAGS.data_dir)
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)
    
    def conv2d_t(x, W):
        return tf.nn.conv2d_transpose(x, W, [FLAGS.batch_size, 32, 32, 3], strides=[1, 1, 1, 1], padding='SAME')
   
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        images_train, _ = inputs(False, data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size) #inputs(eval_data=False, data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
        tf.summary.image("input", images_train, max_outputs=4)        
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(logdir=FLAGS.logdir, graph=tf.get_default_graph())
        batch_max = tf.reduce_max(images_train)
        batch_min = tf.reduce_min(images_train)
        n_images_train = (images_train - batch_min)/(batch_max - batch_min)
        tf.summary.image("norm_input", n_images_train, max_outputs=4)

        #feed input through convolution
        W_conv1 = weight_variable([5, 5, 3, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(n_images_train, W_conv1) + b_conv1)
        print(h_conv1)
        batch_data = tf.reshape(h_conv1, shape=[FLAGS.batch_size, 32*32*16])                
        variable_summaries("TrainingDataStats", batch_data, "batched_data")

        #autoencoder
        dA = AutoEncoder(batch_data, FLAGS.encoder_network, FLAGS.noise_level, True, FLAGS.acitivation_kind, FLAGS.optimizer_kind, FLAGS.learning_rate, global_step=global_step)
        
        #feed decoded info through second convolution
        reconstr = tf.reshape(dA.reconstruction, [tf.shape(dA.reconstruction)[0], 32, 32, 16])        
        W_conv2 = weight_variable([5, 5, 16, 3])
        b_conv2 = bias_variable([3])
        reconstr = tf.nn.relu(conv2d(reconstr, W_conv2) + b_conv2) 
        tf.summary.image("reconstructed_input", reconstr, max_outputs=4)

        n_reconstr = reconstr * (batch_max - batch_min) + batch_min
        tf.summary.image("renormed_reconstructed_input", n_reconstr, max_outputs=4)

        #define new loss
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(reconstr, n_images_train))))       
        tf.summary.scalar("Training loss", loss)
        optimizer_kind = FLAGS.optimizer_kind

        if optimizer_kind.lower() == "momentumoptimizer":
            optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9)
        elif optimizer_kind == "adamoptimizer":
            optimizer = tf.train.AdamOptimizer()
        elif optimizer_kind == "adagradoptimizer":
            optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
        else:
            optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
            
        train_op = optimizer.minimize(loss, global_step=global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.10f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.logdir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()]) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

#raw autoencoding, without convolutions ;-)
def main(_):
    maybe_download_and_extract(FLAGS.data_dir)
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        images_train, _ = inputs(False, data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size) #inputs(eval_data=False, data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
        tf.summary.image("input", images_train, max_outputs=4)        
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(logdir=FLAGS.logdir, graph=tf.get_default_graph())
        batch_max = tf.reduce_max(images_train)
        batch_min = tf.reduce_min(images_train)
        n_images_train = (images_train - batch_min)/(batch_max - batch_min)
        tf.summary.image("norm_input", n_images_train, max_outputs=4)
        batch_data = tf.reshape(images_train, shape=[FLAGS.batch_size, IMAGE_SIZE*IMAGE_SIZE*3])        
        variable_summaries("TrainingDataStats", batch_data, "batched_data")
        dA = AutoEncoder(batch_data, FLAGS.encoder_network, FLAGS.noise_level, True, FLAGS.acitivation_kind, FLAGS.optimizer_kind, FLAGS.learning_rate, global_step=global_step)
        reconstr = tf.reshape(dA.reconstruction, [tf.shape(dA.reconstruction)[0], 32, 32, 3])
        tf.summary.image("reconstructed_input", reconstr, max_outputs=4)
        n_reconstr = reconstr * (batch_max - batch_min) + batch_min
        tf.summary.image("renormed_reconstructed_input", n_reconstr, max_outputs=4)
        loss = dA.loss
        tf.summary.scalar("Training loss", loss)
        train_op = dA.train_op(global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.10f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.logdir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()]) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                

if __name__ == "__main__":
    tf.app.run(main=main)
    #tf.app.run(main=conv_main)