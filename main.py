import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/hidro.csv"

features = tf.placeholder(tf.int32, shape=[6], name='n')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 2  # total classes (0 or 1)

printerop = tf.Print(total, [country, features, total], name='printer')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            n, pessoas_predio, sensor_vazamento, sensor_presenca, segundo_medida, horario_limpeza, dia_util, vazamento = line.strip().split(",")
            # Run the Print ob
            total = sess.run(printerop, feed_dict={ features: [pessoas_predio, sensor_vazamento, sensor_presenca, segundo_medida, horario_limpeza, dia_util], country: n})
            print(n, total)
