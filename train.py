import time

import tensorflow as tf

import utils


class Trainer:
    def __init__(self, model, criterion, optimizer, content_image,
                 style_image, content_weight, style_weight, tv_weight,
                 epochs, len_epoch):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.targets = {'content': model(content_image)['content'],
                        'style': model(style_image)['style']}
        self.loss_weights = {'content': content_weight,
                             'style': style_weight,
                             'tv': tv_weight}
        self.epochs = epochs
        self.len_epoch = len_epoch

    def train(self, image):
        start = time.time()
        step = 0
        for n in range(self.epochs):
            for m in range(self.len_epoch):
                step += 1
                self.train_step(image)
                print(".", end='')
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end-start))

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.model(image)
            loss = self.criterion(self.targets, outputs, self.loss_weights)
            loss += self.loss_weights['tv'] * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(grad, image)])
        image.assign(utils.clip_0_1(image))
