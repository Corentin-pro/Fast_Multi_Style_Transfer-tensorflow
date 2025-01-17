import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import tensorflow.compat.v1 as tf1

from src.layers import conv_layer, conv_tranpose_layer, pooling, residual_block


class EngineMultiStyle:
    def __init__(self, tf_session: tf1.Session, content_data_size: int, checkpoint_path: str,
                 style_count=16, style_control=None, resize=None, broadcast_style=None, hidden_out=False):
        self.tf_session = tf_session
        self.content_data_szie = content_data_size
        self.checkpoint_path = checkpoint_path

        if broadcast_style is not None:
            self.image_placeholder = tf1.placeholder(
                tf.uint8, shape=[content_data_size, content_data_size, 3],
                name='image_placeholder')
            image_input = tf.broadcast_to(
                self.image_placeholder,
                [broadcast_style, content_data_size, content_data_size, 3])
            self.style_placeholder = tf1.placeholder(
                tf.float32, shape=[None, style_count], name='style_placeholder')
        elif style_control is None:
            self.image_placeholder = tf1.placeholder(
                tf.uint8, shape=[None, content_data_size, content_data_size, 3],
                name='image_placeholder')
            image_input = self.image_placeholder
            self.style_placeholder = tf1.placeholder(
                tf.float32, shape=[None, style_count], name='style_placeholder')
        else:
            self.image_placeholder = tf1.placeholder(
                tf.uint8, shape=[len(style_control), content_data_size, content_data_size, 3], name='image_placeholder')
            image_input = self.image_placeholder
            self.style_placeholder = tf1.placeholder(
                tf.float32, shape=[len(style_control), style_count], name='style_placeholder')
        self.style_control = style_control

        self.network = self.mst_net(
            tf.cast(image_input, tf.float32),
            style_control=self.style_placeholder if style_control is None else self.style_control)
        # self.output = tf.minimum(tf.maximum(self.network, 0), 255)
        self.output = self.network

        if hidden_out:
            def get_features(tensor_name: str, output_name: str, channel_count: int):
                with tf.variable_scope('Normalization'):
                    output = tf_session.graph.get_tensor_by_name(tensor_name)[1:5, :, :, :channel_count]
                    means, variances = tf.nn.moments(output, [0, 1, 2])
                    output = tf.nn.batch_normalization(output, means, variances, 0.5, 1, 1e-6)
                    output = tf.clip_by_value(output, 0, 1)
                return tf.cast(output * 200.0, tf.uint8, name=output_name)

            get_features('conv3/Relu:0', 'input_features_1', 3)
            get_features('res2_a/conv/Relu:0', 'input_features_2', 6)

            with tf.variable_scope('Normalization'):
                res_conv3 = tf_session.graph.get_tensor_by_name('res3_a/conv/Relu:0')
                res_conv3 = tf.reshape(
                    tf1.nn.avg_pool2d(res_conv3[:1], 10, 10, 'VALID'),
                    [(6 * (content_data_size // 256)) * (6 * (content_data_size // 256)) * 128])
                means, variances = tf.nn.moments(res_conv3, [0])
                res_conv3 = tf.nn.batch_normalization(res_conv3, means, variances, 0.5, 1, 1e-6)
                res_conv3 = tf.round(tf.clip_by_value(res_conv3, 0, 1))
            res_conv3 = tf.cast(res_conv3, dtype=tf.int32, name='sub_latent')
            prev_con3 = tf.placeholder(tf.int32, shape=res_conv3.shape, name='prev_latent')
            res_conv3_change = tf.square(prev_con3 - res_conv3)
            tf.cast(res_conv3_change, dtype=tf.uint8, name='latent_change')

            get_features('res4_a/conv/Relu:0', 'output_features_1', 3)
            get_features('res5_b/add:0', 'output_features_2', 6)

        if resize:
            self.output = tf1.image.resize_bilinear(self.output, (resize, resize))
        self.output = tf.cast(self.output, tf.uint8, name='output')

        train_writer = tf.summary.FileWriter('engine', self.tf_session.graph, flush_secs=20)

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
    parser.add_argument('--dynamic_style', action='store_true', help='Use dynamic style control')
    arguments = parser.parse_args()

    style_control = np.asarray([[float(value) for value in arguments.style] for _ in range(arguments.batch_size)])

    input_image = Image.open(arguments.input_image).convert('RGB')
    input_image = input_image.resize((arguments.input_size, arguments.input_size))
    input_image = np.asarray(input_image)
    input_image = np.asarray([input_image] * arguments.batch_size)

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)

    with tf1.Session(config=session_config).as_default() as session:
        if arguments.dynamic_style:
            engine = EngineMultiStyle(
                session, arguments.input_size, arguments.checkpoint_path, resize=arguments.output_size)
            output = engine.predict(input_image, style_control=style_control)[0]

            start_time = time.time()
            prediction_count = 30
            for _ in range(prediction_count):
                output = engine.predict(input_image, style_control=style_control)[0]
            time_spent = time.time() - start_time
        else:
            engine = EngineMultiStyle(
                session, arguments.input_size, arguments.checkpoint_path,
                style_control=style_control, resize=arguments.output_size)
            output = engine.predict(input_image)[0]

            start_time = time.time()
            prediction_count = 30
            for _ in range(prediction_count):
                output = engine.predict(input_image)[0]
            time_spent = time.time() - start_time

        print('{} predictions in {:.03f}s => {:.02f}FPS ({:.02f} batch/s)'.format(
            prediction_count, time_spent,
            (prediction_count * arguments.batch_size) / time_spent,
            prediction_count / time_spent))

        Image.fromarray(output).save('output.png')


if __name__ == '__main__':
    main()
