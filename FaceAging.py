import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import time
from PIL import Image
from glob import glob

# Define the FaceAging class
class FaceAging:
    def __init__(self, checkpoint_dir, sample_dir, data_dir, image_size, batch_size, num_z_channels, num_categories, epochs, learning_rate, beta1, is_crop):
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.is_crop = is_crop

        # Build the generator and discriminator models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # Define loss functions and optimizers
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Initialize optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)

        # Define a checkpoint manager to save and restore models
        self.checkpoint = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=5)

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(128, input_shape=(self.num_z_channels + self.num_categories,)))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(256))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(512))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dense(3 * self.image_size * self.image_size, activation='tanh'))
        model.add(layers.Reshape((self.image_size, self.image_size, 3)))
        return model

    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(self.image_size, self.image_size, 3)))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def train_step(self, images, batch_z, batch_category):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(tf.concat([batch_z, batch_category], axis=1), training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self):
        # Load and preprocess your dataset (replace with your dataset loading code)
        data = glob(os.path.join(self.data_dir, "*.jpg"))
        np.random.shuffle(data)
        ims = data[:self.batch_size]

        for epoch in range(self.epochs):
            batch_idxs = len(ims) // self.batch_size

            for idx in range(0, batch_idxs):
                batch_files = ims[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch = [self.get_image(batch_file, self.image_size, self.is_crop) for batch_file in batch_files]

                # Preprocess your batch and create batch_z and batch_category
                batch_images = np.array(batch).astype(np.float32)  # Preprocess your images
                batch_z = np.random.normal(0, 1, (self.batch_size, self.num_z_channels)).astype(np.float32)
                batch_category = np.zeros((self.batch_size, self.num_categories))
                batch_category[:, idx % self.num_categories] = 1.0  # Adjust this based on your category labels

                # Train the model for one batch
                gen_loss, disc_loss = self.train_step(batch_images, batch_z, batch_category)

                if idx % 100 == 0:
                    print(f'Epoch: [{epoch}/{self.epochs}] Batch: {idx}/{batch_idxs} Gen Loss: {gen_loss}, Disc Loss: {disc_loss}')

            # Save checkpoints and sample images
            if (epoch + 1) % 5 == 0:
                self.checkpoint_manager.save()
                self.sample_images(epoch)

    def sample_images(self, epoch):
        # Generate and save sample images
        num_samples = 1  # Adjust the number of samples you want to generate
        sample_z = np.random.normal(0, 1, (num_samples, self.num_z_channels)).astype(np.float32)
        for i in range(self.num_categories):
            sample_category = np.zeros((num_samples, self.num_categories))
            sample_category[:, i] = 1.0

            generated_images = self.generator(tf.concat([sample_z, sample_category], axis=1), training=False)

            for j in range(num_samples):
                image = Image.fromarray((generated_images[j].numpy() * 127.5 + 127.5).astype(np.uint8))
                image.save(os.path.join(self.sample_dir, f'epoch{epoch}_sample{i}_sample{j}.png'))

    def get_image(self, image_path, image_size, is_crop):
        # Replace this with your image loading and preprocessing code
        # You can use libraries like OpenCV or PIL to load and preprocess images
        return processed_image

if __name__ == "__main__":
    checkpoint_dir = 'C:\\Users\\halwa\\OneDrive\\Desktop\\FaceAging\\Project\\Age Progression by Conditional Adversarial Autoencoder\\data'
    sample_dir = 'C:\\Users\\halwa\\OneDrive\\Desktop\\FaceAging\\Project\\Age Progression by Conditional Adversarial Autoencoder\\data'
    data_dir = 'C:\\Users\\halwa\\OneDrive\\Desktop\\FaceAging\\Project\\Age Progression by Conditional Adversarial Autoencoder\\data'
    image_size = 64
    batch_size = 64
    num_z_channels = 100
    num_categories = 5  # Modify based on your category labels
    epochs = 100
    learning_rate = 0.0002
    beta1 = 0.5
    is_crop = True

    face_aging_model = FaceAging(checkpoint_dir, sample_dir, data_dir, image_size, batch_size, num_z_channels, num_categories, epochs, learning_rate, beta1, is_crop)
    face_aging_model.train()