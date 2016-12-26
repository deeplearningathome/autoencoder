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
import time
#these are helper functions to read mnist data. They are part of Tensorflow models
from utils import maybe_download, extract_data, extract_labels, variable_summaries
from tensorflow.contrib.tensorboard.plugins import projector

flags = tf.flags
flags.DEFINE_string("encoder_network", "784,128,10", "specifies encoder network")
flags.DEFINE_float("noise_level", 0.0, "noise level for denoising autoencoder")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_epochs", 60, "number of epochs")
flags.DEFINE_integer("eval_every_step", 2000, "evaluate every x steps")
flags.DEFINE_string("acitivation_kind", "sigmoid", "type of neuron activations")
flags.DEFINE_string("learning_rate", 0.1, "learning rate")
flags.DEFINE_string("optimizer_kind", "rmsprop", "type of oprtimizer")
flags.DEFINE_string("logdir", "tblogs", "tensorboard logs")
FLAGS = flags.FLAGS

def main(_):
    VALIDATION_SIZE = 5000 # Size of the MNIST validation set.    
    """
    This is Mnist specific example
    """
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    VALIDATION_SIZE = 5000  # Size of the validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    train_size = train_labels.shape[0]
    #(self, layers, noise, batch_size, is_training, activation='sigmoid')
    dA = AutoEncoder(FLAGS.encoder_network, FLAGS.noise_level,
        True, FLAGS.acitivation_kind, FLAGS.optimizer_kind, FLAGS.learning_rate)
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(logdir=FLAGS.logdir, graph=tf.get_default_graph())
    #adding summaries for Tensorboard    
    tf.summary.image("input", tf.reshape(dA.x, [tf.shape(dA.x)[0], 28, 28, 1]), max_outputs=4)
    tf.summary.image("reconstructed_input", tf.reshape(dA.reconstruction, [tf.shape(dA.reconstruction)[0], 28, 28, 1]), max_outputs=4)
    variable_summaries("encodings", dA.encoding, 'embedding')
    eval_loss = dA.loss
    tf.summary.scalar("Evaluation Loss", eval_loss)
    merged = tf.summary.merge_all()
    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in xrange(int(FLAGS.num_epochs * train_size) // FLAGS.batch_size):
             offset = (step * FLAGS.batch_size) % (train_size - FLAGS.batch_size)
             batch_data = train_data[offset:(offset + FLAGS.batch_size), ...].reshape(FLAGS.batch_size, 784)
             feed_dict = {dA.x: batch_data}
             if step % FLAGS.eval_every_step == 0:
                tloss, _ = sess.run([dA.loss, dA.train_op], feed_dict=feed_dict)
                print('Train loss: %.4f' % (tloss))
             else:
                sess.run([dA.train_op], feed_dict=feed_dict)             
             #print('Training minibatch at step %d loss: %.6f' % (step, loss))
             if step % FLAGS.eval_every_step == 0:
                 with tf.name_scope("Validation"):
                    eval_feed_dict = {dA.x: validation_data.reshape(VALIDATION_SIZE, 784)}                    
                    eloss, emerged = sess.run([eval_loss, merged], eval_feed_dict)
                    summary_writer.add_summary(emerged, step)
                    saver.save(sess, FLAGS.logdir + "/model")
                 print('Validation loss: %.4f' % (eloss))
        print('Calculating test error')
        tfeed_dict = {dA.x: test_data.reshape(10000, 784)}
        tloss = sess.run(dA.loss, feed_dict=tfeed_dict)
        print('Test loss: %.6f' %(tloss))

if __name__ == "__main__":
    tf.app.run(main=main)
