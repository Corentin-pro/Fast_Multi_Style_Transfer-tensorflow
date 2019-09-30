import os

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from src.layers import conv_layer, conv_tranpose_layer, pooling, residual_block


class EngineMultiStyle:
    def __init__(self, tf_session: tf1.Session, content_data_size: int, checkpoint_path: str,
                 resize=None):
        self.tf_session = tf_session
        self.content_data_szie = content_data_size
        self.checkpoint_path = checkpoint_path

        self.image_placeholder = tf1.placeholder(
            tf.uint8, shape=[None, content_data_size, content_data_size, 3], name='img')
        self.style_placeholder = tf1.placeholder(
            tf.float32, shape=[None, 16], name='style_placeholder')

        self.network = self.mst_net(
            tf.cast(self.image_placeholder, tf.float32),
            style_control=self.style_placeholder)
        # self.output = tf.minimum(tf.maximum(self.network, 0), 255)
        self.output = self.network

        if resize:
            self.output = tf1.image.resize_bilinear(self.output, (resize, resize))
        self.output = tf.cast(self.output, tf.uint8)

        # train_writer = tf.compat.v1.summary.FileWriter('engine', self.tf_session.graph, flush_secs=20)

        self.saver = tf1.train.Saver(var_list=tf1.trainable_variables())
        self.saver.restore(self.tf_session, self.checkpoint_path)

    def mst_net(self, x, style_control=None, reuse=False):
        with tf1.variable_scope(tf1.get_variable_scope(), reuse=reuse):
            # batch_size, height, width, channels = x.get_shape().as_list()

            x = conv_layer(x, 32, 9, 1, style_control=style_control, name='conv1')
            x = conv_layer(x, 64, 3, 2, style_control=style_control, name='conv2')
            x = conv_layer(x, 128, 3, 2, style_control=style_control, name='conv3')
            x = residual_block(x, 3, style_control=style_control, name='res1')
            x = residual_block(x, 3, style_control=style_control, name='res2')
            x = residual_block(x, 3, style_control=style_control, name='res3')
            x = residual_block(x, 3, style_control=style_control, name='res4')
            x = residual_block(x, 3, style_control=style_control, name='res5')
            x = conv_tranpose_layer(x, 64, 3, 2, style_control=style_control, name='up_conv1',
                                    engine=True)
            x = pooling(x)
            x = conv_tranpose_layer(x, 32, 3, 2, style_control=style_control, name='up_conv2',
                                    engine=True)
            x = pooling(x)
            x = conv_layer(x, 3, 9, 1, relu=False, style_control=style_control, name='output')
            preds = tf.nn.tanh(x) * 127 + 128
        return preds

    def predict(self, images, style_control=None):
        if style_control is None:
            return self.tf_session.run(
                self.output, feed_dict={self.image_placeholder: images})
        return self.tf_session.run(
            self.output, feed_dict={
                self.image_placeholder: images,
                self.style_placeholder: style_control})

def main():
    import argparse
    import time

    from PIL import Image
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', help='Path to image to parse')
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size to use')
    parser.add_argument('--style', nargs='+',
                        default=[1.] * 16, help='List of weights for style')
    parser.add_argument('--input-size', type=int, default=256, help='Shape of input to use (depends on checkpoint)')
    parser.add_argument('--output-size', type=int, default=None, help='Shape of output image')
    arguments = parser.parse_args()

    style_control = np.asarray([[float(value) for value in arguments.style] for _ in range( arguments.batch_size)])

    input_image = Image.open(arguments.input_image).convert('RGB')
    input_image = input_image.resize((arguments.input_size, arguments.input_size))
    input_image = np.asarray(input_image)
    input_image = np.asarray([input_image] * arguments.batch_size)

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)

    with tf1.Session(config=session_config).as_default() as session:
        engine = EngineMultiStyle(session, arguments.input_size, arguments.checkpoint_path, resize=arguments.output_size)
        output = engine.predict(input_image, style_control=style_control)[0]

        start_time = time.time()
        prediction_count = 30
        for _ in range(prediction_count):
            output = engine.predict(input_image, style_control=style_control)[0]
        time_spent = time.time() - start_time

        print('{} predictions in {:.03f}s => {:.02f}FPS ({:.02f} batch/s)'.format(
            prediction_count, time_spent,
            (prediction_count * arguments.batch_size) / time_spent,
            prediction_count / time_spent))

        Image.fromarray(output).save('output.png')


if __name__ == '__main__':
    main()
