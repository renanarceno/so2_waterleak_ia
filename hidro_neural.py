import tensorflow as tf


# import random as r


# Seleciona os dados para treino
def get_train_batch():
    return get_batch(filename="hidrosimulation_train.csv")


def get_test_batch():
    return get_batch(filename="hidrosimulation_test.csv")


def get_batch(filename):
    xt = list()
    yt = list()
    with open(filename) as file:
        # Pular cabeçalho
        next(file)
        for line in file:
            pessoas, maquinas, vazao_total, vazao_1, vazao_2, vazao_3, vazamento, normal = line.strip().split(",")
            xt.append([float(pessoas), float(maquinas), float(vazao_total), float(vazao_1), float(vazao_2), float(vazao_3)])
            yt.append([int(vazamento), int(normal)])
    return xt, yt


"""
    Implementando a regressão
"""
# Placeholder não é um valor específico, e sim um "espaço reservado" para o framework usar para a computação
x = tf.placeholder(tf.float32, [None, 6])

# Variable é um "tensor" modificado que está inserido dentro do graph do TensorFlow, usado pela computação da rede
W = tf.Variable(tf.zeros([6, 2]))
b = tf.Variable(tf.zeros([2]))

# Utilizaremos uma rede neural utilizando ativações "softmax"
y = tf.nn.softmax(tf.matmul(x, W) + b)

"""
    Implementando o treinamento
"""
# Para treinar nossa rede neural, precisamos definir o significado de uma "rede boa".
# Uma forma de definirmos quão boa é uma rede, é calculando o oposto: quão ruim ela é.
# Isso é chamado de custo ou perda e representa quão longe do resultado esperado nossa rede está
# Uma forma bem comum de determinarmos a perda de um modelo é chamado de "cross-entropy" e é definido pela função:
# Hy'(y) = -SOMATORIO (yi' * log(yi))
# Implementando "cross-entropy":
# 1o: Criamos os placeholders que abrigarão os valores y
y_ = tf.placeholder(tf.float32, [None, 2])

# 2o: implementamos a função de entropia
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
# NÃO USAR DESSA FORMA, USAR: tf.nn.softmax_cross_entropy_with_logits

# Criando configurações do treinamento
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = get_train_batch()
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Testa entradas e retorna um resultado
prediction = sess.run(y, feed_dict={x: [[0., 0., 0., 0., 0., 0.]]})
print("Prediction: " + str(prediction))
xtest_batch, ytest_batch = get_test_batch()
print(sess.run(accuracy, feed_dict={x: xtest_batch, y_: ytest_batch}))
