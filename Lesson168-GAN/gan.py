import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


tf.random.set_seed(12345)


class Generator(keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        # [z, 100] --> [z, 3*3*512]
        self.fc = layers.Dense(3*3*512)

        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        # [z, 100] --> [z, 3*3*512]
        x = self.fc(inputs)
        # [z, 3*3*512] --> [z, 3, 3, 512]
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = tf.nn.leaky_relu(self.bn1(self.conv1(x)))

        x = tf.nn.leaky_relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)

        x = tf.nn.tanh(x)

        return x


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        # [b, 64, 64, c] --> [b, 1
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')

        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # [b, h, w, c] --> [b, -1]
        self.flatten = layers.Flatten()

        # [b, -1] --> [b, 1]
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):

        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        x = self.flatten(x)

        logist = self.fc(x)

        return logist



def main():

    g = Generator()
    d = Discriminator()

    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])

    x_hat = d(x)
    print(x_hat)

    prob = g(z)
    print(prob.shape)





if __name__ == '__main__':
    main()