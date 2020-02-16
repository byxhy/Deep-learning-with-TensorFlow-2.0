import tensorflow as tf
import numpy as np
import os
import glob
# from scipy.misc import toimage
from PIL import Image

from gan import Generator, Discriminator
from dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)

    # toimage(final_image).save(image_path)
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1. treat real image as real
    # 2. treat fake image as fake

    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    loss = d_loss_real + d_loss_fake

    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 1. treat fake image as real

    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)

    loss = celoss_ones(d_fake_logits)

    return loss


def main():
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate N * 1GB of memory on the first GPU
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)


    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # hyper parameters
    z_dim = 100
    batch_size = 64
    lr = 0.002
    is_training = True

    img_path = glob.glob('./faces/*.jpg')
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(len(img_path))
    print(dataset, img_shape)

    # sample = next(iter(dataset))
    #
    # print(img_path)
    # print(len(img_path))
    # print(sample.shape)
    # print(tf.reduce_min(sample), tf.reduce_max(sample))

    dataset = dataset.repeat() # It probably useful for your fuzzy dataset
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    generator.summary()

    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))
    discriminator.summary()

    g_optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.5)


    for epoch in range(4000):

        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)

        # train Discriminator
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)

        grads_dis = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads_dis, discriminator.trainable_variables))

        # train Generator
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)

        grads_gen = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))


        if epoch % 100 == 0:
            print('Epoch', epoch, ' d_loss:', float(d_loss), ' g_loss:', float(g_loss))

            z_fake = tf.random.uniform([100, z_dim])
            fake_image = generator(z_fake, training=False)
            img_save_path = os.path.join('./images/', 'gan-%d.png'%epoch)
            save_result(fake_image.numpy(), 10, img_save_path, color_mode='P')

            generator.save_weights('./save_weights/generator.ckpt')
            discriminator.save_weights('./save_weights/discriminator.ckpt')



if __name__ == '__main__':
    main()