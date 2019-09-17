import os

import tensorflow as tf
import tensorflow.compat.v1 as tf1

from src.layers import conv_layer, conv_tranpose_layer, pooling, residual_block


class Engine:
    def __init__(self, tf_session: tf.Session, style_control: list,
                 content_data_size: int, checkpoint_path: str):
        self.tf_session = tf_session
        self.content_data_szie = content_data_size
        self.checkpoint_path = checkpoint_path
        self.style_control = style_control

        self.image_placeholder = tf1.placeholder(
            tf.float32, shape=[None, content_data_size, content_data_size, 3], name='img')

        self.network = self.mst_net(self.image_placeholder, style_control=style_control)

        self.saver = tf1.train.Saver(var_list=tf.trainable_variables())
        self.saver.restore(self.tf_session, self.checkpoint_path)

    def mst_net(self, x, style_control=None, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
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
            preds = tf.nn.tanh(x) * 150 + 255. / 2
        return preds

    def predict(self, images):
        return self.tf_session.run(self.network, feed_dict={self.image_placeholder: images})


def main():
    import argparse
    import time

    from PIL import Image
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('input_image', help='Path to image to parse')
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--style', nargs='+',
                        default=[1.] * 16, help='List of weights for style')
    parser.add_argument('--input_size', type=int, default=256, help='Shape of input to use (depends on checkpoint)')
    arguments = parser.parse_args()

    style_control = [float(value) for value in arguments.style]

    input_image = Image.open(arguments.input_image).convert('RGB')
    input_image = input_image.resize((arguments.input_size, arguments.input_size))
    input_image = np.asarray(input_image)

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)

    with tf1.Session(config=session_config).as_default() as session:
        engine = Engine(session, style_control, arguments.input_size, arguments.checkpoint_path)

        output = engine.predict([input_image])[0]

        start_time = time.time()
        prediction_count = 30
        for _ in range(prediction_count):
            output = engine.predict([input_image])[0]
        time_spent = time.time() - start_time
        print('{} predictions in {:.03f}s => {}FPS'.format(
            prediction_count, time_spent, prediction_count / time_spent))

        Image.fromarray(output.astype(np.uint8)).save('output.png')


if __name__ == '__main__':
    main()
