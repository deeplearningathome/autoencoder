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

class AutoEncoder(object):
    """
    Simple de-noising autoencoder
    """
    def __init__(self, input, layers, noise, is_training, activation='sigmoid', optimizer_kind='rmsprop', lr=0.01, global_step=None):
        """
        Layers should define encoder topology. Decoder's topology is inferred from that.abs
        First number in layers should be feature size. Last number - size of encoding
        """
        self._global_step = global_step
        self._activation = activation
        layers_arr = [int(layer) for layer in layers.split(',')]
        assert len(layers_arr) >= 2
        self._feature_size = layers_arr[0]
        self._encoding_size = layers_arr[-1]
        self._x = input#tf.placeholder(tf.float32, name='x', shape=[None, self._feature_size])
        #create encoder
        self._encoding_matrices = []
        self._encoding_biases = []
        #for constrained autoencoder we don't have decoding matrices
        #since they are just equal to transposed encoding matrices
        self._decoding_biases = []
        in_size = self._feature_size
        ind = 0
        with tf.variable_scope("ConstrainedAutoEncoder"):
            initializer = tf.contrib.layers.xavier_initializer()
            for out_size in layers_arr[1:]:
                self._encoding_matrices.append(
                    tf.get_variable("W" + str(ind), shape=[in_size, out_size]))
                self._encoding_biases.append(tf.get_variable(
                    "be" + str(ind), shape=[out_size], initializer=initializer))
                self._decoding_biases.append(tf.get_variable(
                    "bd" + str(ind), shape=[in_size], initializer=initializer))
                ind += 1
                in_size = out_size
            if noise > 0.0:
                self._e = self.encode(self._x)
            else:
                self._e = self.encode(tf.nn.dropout(self._x, 1 - noise))
            self._z = self.decode(self._e)
            self._loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self._x, self._z))))
        if is_training:
            if optimizer_kind.lower() == "momentumoptimizer":
                optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
            elif optimizer_kind == "adamoptimizer":
                optimizer = tf.train.AdamOptimizer()
            elif optimizer_kind == "adagradoptimizer":
                optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
            self._train_op = optimizer.minimize(self.loss, global_step=global_step)

    def encode(self, x):
        with tf.name_scope("ConstrainedAutoEncoder_Encoder"):
            inpt = x
            for i in range(0, len(self._encoding_matrices)):
                W = self._encoding_matrices[i]
                b = self._encoding_biases[i]
                logits = tf.nn.bias_add(tf.matmul(inpt, W), b)
                if self._activation == "relu":
                    inpt = tf.nn.relu(logits, name="Embeddings")
                else:
                    inpt = tf.sigmoid(logits, name="Embeddings")
            return inpt

    def decode(self, encoding):
        with tf.name_scope("ConstrainedAutoEncoder_Decoder"):
            inpt = encoding
            for i in range(len(self._encoding_matrices)-1, -1, -1):
                Wd = tf.transpose(self._encoding_matrices[i])
                bd = self._decoding_biases[i]
                logits = tf.nn.bias_add(tf.matmul(inpt, Wd), bd)
                if self._activation == "relu":
                    inpt = tf.nn.relu(logits)
                else:
                    inpt = tf.sigmoid(logits)
            return inpt

    @property
    def encoding(self):
        return self._e

    @property
    def reconstruction(self):
        return self._z

    #@property
    def x(self):
        return self._x

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def loss(self):
        return self._loss

    def train_op(self, global_step=None):
        if global_step:
            self._global_step = global_step
        return self._train_op
