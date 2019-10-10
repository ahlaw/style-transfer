import argparse

import tensorflow as tf

from model.style_transfer import StyleTransfer, style_content_loss
from train import Trainer
import utils


# default arguments
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
TV_WEIGHT = 30
LEARNING_RATE = 0.02
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 100


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--content', type=str,
                        dest='content_path',
                        help='content image path',
                        required=True)
    parser.add_argument('--style', type=str,
                        dest='style_path',
                        help='style image path',
                        required=True)
    parser.add_argument('--output', type=str,
                        dest='dest',
                        help='output path',
                        required=True)
    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight',
                        default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight',
                        default=STYLE_WEIGHT)
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation weight',
                        default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate',
                        default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int,
                        dest='epochs',
                        help='number of epochs',
                        default=NUM_EPOCHS)
    parser.add_argument('--len-epoch', type=int,
                        dest='len_epoch',
                        help='steps per epoch',
                        default=STEPS_PER_EPOCH)

    args = parser.parse_args()

    content_image = utils.load_image(args.content_path)
    style_image = utils.load_image(args.style_path)

    content_layers = ['block5_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    model = StyleTransfer(content_layers, style_layers)

    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate,
                                   beta_1=0.99, epsilon=1e-1)

    trainer = Trainer(model, style_content_loss, optimizer, content_image,
                      style_image, args.content_weight, args.style_weight,
                      args.tv_weight, args.epochs, args.len_epoch)

    image = tf.Variable(content_image)
    trainer.train(image)

    utils.tensor_to_image(image).save(args.dest)


if __name__ == '__main__':
    main()
