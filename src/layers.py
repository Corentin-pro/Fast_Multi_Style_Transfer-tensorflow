import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf
import tensorflow.compat.v1 as tf1


def conv_layer(net, num_filters, filter_size, strides, style_control=None, relu=True, name='conv'):
    with tf1.variable_scope(name):
        _, _, _, channels = net.get_shape().as_list()
        weights_shape = [filter_size, filter_size, channels, num_filters]
        weights_init = tf1.get_variable(
            name, shape=weights_shape, initializer=tf.truncated_normal_initializer(stddev=.01))
        strides_shape = [1, strides, strides, 1]

        p = (filter_size - 1) // 2
        if strides == 1:
            net = tf.pad(net, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding="VALID")
        else:
            net = tf.nn.conv2d(net, weights_init, strides_shape, padding="SAME")

        net = conditional_instance_norm(net, style_control=style_control)
        if relu:
            net = tf.nn.relu(net)

    return net


def conv_tranpose_layer(net, num_filters, filter_size, strides, style_control=None, name='conv_t',
                        engine=False):
    with tf1.variable_scope(name):
        b, w, h, c = net.get_shape().as_list()
        weights_shape = [filter_size, filter_size, num_filters, c]
        weights_init = tf1.get_variable(
            name, shape=weights_shape, initializer=tf.truncated_normal_initializer(stddev=.01))

        batch_size, rows, cols, _ = [i.value for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        if engine:
            batch_size = tf.shape(net)[0]
        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, strides, strides, 1]

        p = (filter_size - 1) // 2
        if strides == 1:
            net = tf.pad(net, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding="VALID")
        else:
            net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding="SAME")
        net = conditional_instance_norm(net, style_control=style_control)

    return tf.nn.relu(net)


def residual_block(net, filter_size=3, style_control=None, name='res'):
    with tf1.variable_scope(name+'_a'):
        tmp = conv_layer(net, 128, filter_size, 1, style_control=style_control)
    with tf1.variable_scope(name+'_b'):
        output = net + conv_layer(tmp, 128, filter_size, 1, style_control=style_control, relu=False)
    return output


def conditional_instance_norm(net, style_control=None, name='cond_in'):
    with tf1.variable_scope(name):
        _, _, _, channels = [i.value for i in net.get_shape()]
        mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

        var_shape = [channels]

        # Hard coded style
        if isinstance(style_control, (list, np.ndarray)):
            shift = {}
            scale = {}
            # Single style per batch
            if len(np.asarray(style_control).shape) == 1:
                for style_index, style_weight in enumerate(style_control):
                    if style_weight == 0 or style_index in shift:
                        continue
                    with tf1.variable_scope('{0}'.format(style_index) + '_style'):
                        shift[style_index] = tf1.get_variable(
                            'shift', shape=var_shape, initializer=tf.constant_initializer(0.))
                        scale[style_index] = tf1.get_variable(
                            'scale', shape=var_shape, initializer=tf.constant_initializer(1.))
                epsilon = 1e-3
                normalized = tf.subtract(net, mu) / tf.sqrt(sigma_sq + epsilon)

                style_scale = None
                style_shift = None
                for style_index, style_weight in enumerate(style_control):
                    if style_weight == 0:
                        continue
                    if style_scale is None:
                        style_scale = scale[style_index] * style_weight
                    else:
                        style_scale = tf.add(style_scale, scale[style_index] * style_weight)
                    if style_shift is None:
                        style_shift = shift[style_index] * style_weight
                    else:
                        style_shift = tf.add(style_shift, shift[style_index] * style_weight)
                style_scale = style_scale / sum(style_control)
                style_shift = style_shift / sum(style_control)

                output = style_scale * normalized + style_shift
            else:  # Multiple style per batch
                for batch_index, batch_style in enumerate(style_control):
                    for style_index, style_weight in enumerate(batch_style):
                        if style_weight == 0 or style_index in shift:
                            continue
                        with tf1.variable_scope('{0}'.format(style_index) + '_style'):
                            shift[style_index] = tf1.get_variable(
                                'shift', shape=var_shape, initializer=tf.constant_initializer(0.))
                            scale[style_index] = tf1.get_variable(
                                'scale', shape=var_shape, initializer=tf.constant_initializer(1.))
                epsilon = 1e-3
                normalized = tf.subtract(net, mu) / tf.sqrt(sigma_sq + epsilon)

                style_scales = []
                style_shifts = []
                for batch_index, batch_style in enumerate(style_control):
                    style_scale = None
                    style_shift = None
                    for style_index, style_weight in enumerate(batch_style):
                        if style_weight == 0:
                            continue
                        if style_scale is None:
                            style_scale = scale[style_index] * style_weight
                        else:
                            style_scale = tf.add(style_scale, scale[style_index] * style_weight)
                        if style_shift is None:
                            style_shift = shift[style_index] * style_weight
                        else:
                            style_shift = tf.add(style_shift, shift[style_index] * style_weight)
                    style_scales.append(style_scale / sum(style_control[batch_index]))
                    style_shifts.append(style_shift / sum(style_control[batch_index]))

                style_scales = tf.expand_dims(tf.expand_dims(tf.stack(style_scales), axis=1), axis=1)
                style_shifts = tf.expand_dims(tf.expand_dims(tf.stack(style_shifts), axis=1), axis=1)
                output = style_scales * normalized + style_shifts
        # Single style per batch
        elif style_control.shape.as_list()[0] is not None:
            shift = []
            scale = []
            strided_style = tf.unstack(style_control)
            for i in range(style_control.shape.as_list()[0]):
                with tf1.variable_scope('{0}'.format(i) + '_style'):
                    style_shift = tf1.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.))
                    shift.append(style_shift * strided_style[i])
                    style_scale = tf1.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.))
                    scale.append(style_scale * strided_style[i])
            epsilon = 1e-3
            normalized = tf.subtract(net, mu) / tf.sqrt(sigma_sq + epsilon)

            style_shifts = tf.stack(shift)
            style_scales = tf.stack(scale)

            style_shift = tf.reduce_sum(style_shifts, axis=0) / tf.reduce_sum(style_control, axis=0)
            style_scale = tf.reduce_sum(style_scales, axis=0) / tf.reduce_sum(style_control, axis=0)
            output = style_scale * normalized + style_shift
        # Multi style
        else:
            shift = []
            scale = []
            strided_style = tf.unstack(style_control, axis=1)
            style_count = style_control.shape.as_list()[1]
            for i in range(style_count):
                with tf1.variable_scope('{0}'.format(i) + '_style'):
                    style_shift = tf1.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.))
                    channel_count = style_shift.shape.as_list()[0]
                    batch_control = tf.broadcast_to(
                        tf.ones(channel_count),
                        [tf.shape(style_control)[0], channel_count]) * tf.expand_dims(strided_style[i], 1)
                    shift.append(style_shift * batch_control)

                    style_scale = tf1.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.))
                    scale.append(style_scale * batch_control)
            epsilon = 1e-3
            normalized = tf.subtract(net, mu) / tf.sqrt(sigma_sq + epsilon)

            style_shifts = tf.stack(shift)
            style_scales = tf.stack(scale)

            style_shift = tf.expand_dims(tf.expand_dims(
                tf.reduce_sum(style_shifts, axis=0) / tf.reduce_sum(style_control, axis=1, keepdims=True),
                axis=1), axis=1)
            style_scale = tf.expand_dims(tf.expand_dims(
                tf.reduce_sum(style_scales, axis=0) / tf.reduce_sum(style_control, axis=1, keepdims=True),
                axis=1), axis=1)
            output = style_scale * normalized + style_shift

    return output


def instance_norm(net, train=True, name='in'):
    with tf1.variable_scope(name):
        _, _, _, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf1.get_variable('shift', shape=var_shape, initializer=tf.constant_initializer(0.))
        scale = tf1.get_variable('scale', shape=var_shape, initializer=tf.constant_initializer(1.))
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def pooling(input):
    return tf1.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 1, 1, 1), padding='SAME')


def total_variation(preds):
     # total variation denoising
     b,w,h,c = preds.get_shape().as_list()
     y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:w-1,:,:])
     x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:h-1,:])
     tv_loss = 2*(x_tv + y_tv)/b/w/h/c
     return tv_loss


def euclidean_loss(input_, target_):
    b,w,h,c = input_.get_shape().as_list()
    return 2 * tf.nn.l2_loss(input_- target_) / b/w/h/c


def gram_matrix(net):
    b,h,w,c = net.get_shape().as_list()
    feats = tf.reshape(net, (b, h*w, c))
    feats_T = tf.transpose(feats, perm=[0,2,1])
    grams = tf.matmul(feats_T, feats) / h/w/c
    return grams


def style_loss(input_, style_):
    b,h,w,c = input_.get_shape().as_list()
    input_gram = gram_matrix(input_)
    style_gram = gram_matrix(style_)
    return 2 * tf.nn.l2_loss(input_gram - style_gram)/b/c/c

