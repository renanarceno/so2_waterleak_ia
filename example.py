# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    filename = "hidro.csv"

    train_x = list()
    train_y = list()
    test_x = list()
    test_y = list()
    """ 
        Total data in csv file: 6914 
        Total to train: 5530
        Total to check: 1384
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with open(filename) as inf:
            # Skip header
            i = 0
            next(inf)
            for line in inf:
                # Read data, using python, into our features
                pessoas_predio, sensor_vazamento, sensor_presenca, segundo_medida, horario_limpeza, dia_util, \
                vazamento = line.strip().split(",")
                if i % 2 == 0:
                    train_x.append([int(pessoas_predio), int(sensor_vazamento), int(sensor_presenca),
                                    int(segundo_medida), int(horario_limpeza), int(dia_util)])
                    train_y.append([int(vazamento)])
                else:
                    test_x.append([int(pessoas_predio), int(sensor_vazamento), int(sensor_presenca),
                                   int(segundo_medida), int(horario_limpeza), int(dia_util)])
                    test_y.append([int(vazamento)])
                i = i + 1

    return train_x, test_x, train_y, test_y


def main():
    train_x, test_x, train_y, test_y = get_iris_data()

    # Layer's sizes
    x_size = 6  # Number of input nodes
    h_size = 256  # Number of hidden nodes
    y_size = 1  # Number of outcomes 0 or 1

    # Symbols
    x = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.float32, shape=[None, y_size])

    # Weight initializations
    w_1 = tf.Variable((x_size, h_size))

    # Forward propagation
    predict = tf.nn.sigmoid(tf.matmul(x, w_1))

    # Backward propagation
    cost = tf.reduce_mean((y-predict)**2)
    updates = tf.train.GradientDescentOptimizer(10).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates, feed_dict={x: train_x[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={x: train_x, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={x: test_x, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (
            epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    var = sess.run(tf.argmax(x, 1), feed_dict={x: [4, 1, 0, 0, 0, 1]})
    print(var)
    sess.close()


if __name__ == '__main__':
    main()
