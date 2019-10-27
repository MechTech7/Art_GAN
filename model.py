import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import cv2
import os
from tqdm import tqdm

#TODO: upload to GITHUB and get this doing cycles on the desktop

gen_learning_rate = 0.0002
disc_learning_rate = 0.0002

gen_final_filters = 160
disc_input_filters = 40

z_latent_size = 100
epochs = 100
example_shape = (128, 128, 3)
g_beta = 0.5
d_beta= 0.5

#NOTE: the purpose of this model is to create a DCGAN model that I can tweak the architecture of as well as try new things

#NOTE: leaky relu comes with the defeault desired alpha of 0.2

#TODO: Add clipping of weights, training method, 
#https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/

#TODO: add support for custom loss functions
class DCGAN:
    def __init__(self, gen_final_filters, disc_input_filters, z_latent_size, sesh, learning_rate=None, n_critic=None, optimizer=None, loss="wasserstein"):
        self.gen_final_filters = gen_final_filters
        self.disc_input_filters = disc_input_filters
        self.z_latent_size = z_latent_size

        self.n_critic = n_critic #number of discriminator training iterations per

        self.sesh = sesh
        
        loss_functions = {"wasserstein": self.wasserstein_loss, "cross-entropy": self.cross_entropy_loss}
        learning_rates = {"wasserstein": 0.00005, "cross-entropy": 0.0002}
        n_critics = {"wasserstein": 5, "cross-entropy": 1}
        

        self.learning_rate = learning_rate
        if self.learning_rate == None:
            self.learning_rate = learning_rates[loss]
            print("-------------using " + loss + " default learning rate: " + str(self.learning_rate))
        
        self.n_critic = n_critic
        if self.n_critic == None:
            self.n_critic = n_critics[loss]
            print("-------------using " + loss + " default n_critic steps: " + str(self.n_critic))

        optimizers = {"wasserstein": tf.train.RMSPropOptimizer(self.learning_rate), "cross-entropy": tf.train.AdamOptimizer(self.learning_rate)}
        self.optimizer = optimizer
        if self.optimizer == None:
            self.optimizer = optimizers[loss]
            print("-------------using " + loss + " default optimizer: " + str(self.optimizer))


        self.loss_function = loss_functions[loss]
        
        self.latent_variable = tf.placeholder(dtype=tf.float32, shape=[None, self.z_latent_size], name="latent_variable")
        self.real_images = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="real_examples")
        print("latent_variable:", self.latent_variable)
        #Generator-Discriminator argument
        self.fake_images = self.generator(self.latent_variable, reuse=False)
        
        self.disc_fake_logits, self.disc_fake = self.discriminator(self.fake_images, constraint=self.clip_weights, reuse=False)
        self.disc_real_logits, self.disc_real = self.discriminator(self.real_images, reuse=True)

        self.disc_loss, self.gen_loss = self.loss_function(self.disc_fake_logits, self.disc_real_logits)
        
        self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
        self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        
        self.gen_opt = self.optimizer.minimize(self.gen_loss, var_list=self.gen_vars)
        self.disc_opt = self.optimizer.minimize(self.disc_loss, var_list=self.disc_vars)

        self.sesh.run(tf.global_variables_initializer())
        
    def clip_weights(self, x, min_val=-0.01, max_val=0.01):
        return tf.clip_by_value(x, min_val, max_val)
    
    def train_step(self, latent_z, input_batch):
        print("CALLED")
        print("n_critic", self.n_critic)
        for i in range(self.n_critic):
            _, d_loss = self.sesh.run([self.disc_opt, self.disc_loss], feed_dict={self.real_images: input_batch, self.latent_variable: latent_z})


        _, g_loss = self.sesh.run([self.gen_opt, self.gen_loss], feed_dict={self.latent_variable: latent_z})

    def train(self, epochs, file_loader):
        pass
    def cross_entropy_loss(self, disc_fake_logits, disc_real_logits):
        disc_real_target = tf.ones_like(disc_real_logits)
        disc_loss = tf.sigmoid_cross_entropy()
        pass
    def wasserstein_loss(self, disc_fake_logits, disc_real_logits):
        disc_loss = tf.reduce_mean(disc_real_logits) - tf.reduce_mean(disc_fake_logits)
        gen_loss = -tf.reduce_mean(disc_fake_logits)

        return disc_loss, gen_loss
    def generator(self, latent_variable, reuse=False):
        
        with tf.variable_scope("Generator", reuse=tf.AUTO_REUSE):
            layer_0 = tf.layers.dense(latent_variable, units=(4 * 4 * 16 * self.gen_final_filters))
            layer_0 = tf.reshape(layer_0, [-1, 4, 4, 16 * self.gen_final_filters]) #giving individual layers variable names for future adaptability into a progressive model
            layer_0 = tf.layers.batch_normalization(layer_0)

            layer_1 = tf.layers.conv2d_transpose(layer_0, filters=(8 * self.gen_final_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            layer_1 = tf.nn.leaky_relu(layer_1)
            layer_1 = tf.layers.batch_normalization(layer_1)

            layer_2 = tf.layers.conv2d_transpose(layer_1, filters=(4 * self.gen_final_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            layer_2 = tf.nn.leaky_relu(layer_2)
            layer_2 = tf.layers.batch_normalization(layer_2)

            layer_3 = tf.layers.conv2d_transpose(layer_2, filters=(2 * self.gen_final_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            layer_3 = tf.nn.leaky_relu(layer_3)
            layer_3 = tf.layers.batch_normalization(layer_3)

            layer_4 = tf.layers.conv2d_transpose(layer_3, filters=(self.gen_final_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            layer_4 = tf.nn.leaky_relu(layer_4)
            layer_4 = tf.layers.batch_normalization(layer_4)

            op_layer = tf.layers.conv2d_transpose(layer_4, filters=3, kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            op_layer = tf.nn.tanh(op_layer)
        return op_layer

    def discriminator(self, input_data, constraint=None, reuse=False):
        #TODO: add weight clipping to layers using kernel_constraint
        constraint_func = constraint
        def no_func(x):
            return x
        if constraint == None:
            constraint_func = no_func
        with tf.variable_scope("Discriminator", reuse=tf.AUTO_REUSE):
            layer_0 = tf.layers.conv2d(input_data, filters=self.disc_input_filters, kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), kernel_constraint=constraint_func)
            layer_0 = tf.nn.leaky_relu(layer_0)
            layer_0 = tf.layers.batch_normalization(layer_0)

            layer_1 = tf.layers.conv2d(layer_0, filters=(2 * self.disc_input_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), kernel_constraint=constraint_func)
            layer_1 = tf.nn.leaky_relu(layer_1)
            layer_1 = tf.layers.batch_normalization(layer_1)

            layer_2 = tf.layers.conv2d(layer_1, filters=(4 * self.disc_input_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), kernel_constraint=constraint_func)
            layer_2 = tf.nn.leaky_relu(layer_2)
            layer_2 = tf.layers.batch_normalization(layer_2)

            layer_3 = tf.layers.conv2d(layer_2, filters=(8 * self.disc_input_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), kernel_constraint=constraint_func)
            layer_3 = tf.nn.leaky_relu(layer_3)
            layer_3 = tf.layers.batch_normalization(layer_3)

            layer_4 = tf.layers.conv2d(layer_3, filters=(16 * self.disc_input_filters), kernel_size=(5, 5), strides=(2, 2), padding="SAME", kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), kernel_constraint=constraint_func)
            layer_4 = tf.nn.leaky_relu(layer_4)
            layer_4 = tf.layers.batch_normalization(layer_4)

            rs_tensor = tf.layers.flatten(layer_4)
            op_logits = tf.layers.dense(rs_tensor, units=1, kernel_constraint=constraint_func)

            op_layer = tf.nn.sigmoid(op_logits)
        return op_logits, op_layer

    def cross_entrop_loss(self):
        #add cross entropy loss
        pass
    def post_process(input_vector):
        #convert the tanh output of the generator to (0, 255)
        pass
    def pre_process(input_vector):
        #scale the input image from (0, 255) to (-1, 1)
        pass
    
    def train(self, input_data, batch_size, epochs):
        #this will be a description of the training step
        #TODO: add reloading of weights
        with self.session as sesh:
            pass



if __name__ == "__main__":
    dcg = DCGAN(gen_final_filters=160, disc_input_filters=40, z_latent_size=100, sesh=tf.Session())
    real_ex = np.random.uniform(-1.0, 1.0, (10, 128, 128, 3))
    low_lvl = np.random.uniform(-1.0, 1.0, (10, 100))
    dcg.train_step(low_lvl, real_ex)