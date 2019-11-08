import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--input-size', type=int, default=256, help='Shape of input to use (depends on checkpoint)')
    parser.add_argument('--style-count', type=int, default=16, help='Number of style')
    parser.add_argument('--output', default='models', help='Directory to output frozen model')
    arguments = parser.parse_args()

    import tensorflow as tf
    if tf.__version__.split('.')[0] == '2':
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
    import tensorflow.compat.v1 as tf1
    # from engine import Engine
    from engine_multi import EngineMultiStyle

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)

    with tf1.Session(config=session_config).as_default() as session:
        # engine = Engine(session, arguments.input_size, arguments.checkpoint_path)
        engine = EngineMultiStyle(
            session, arguments.input_size, arguments.checkpoint_path, style_count=arguments.style_count,
            broadcast_style=5, hidden_out=True)

        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        trained_graph = convert_variables_to_constants(
            session, session.graph_def, ['input_features', 'latent', 'output_features', 'output_1'])
        tf.train.write_graph(trained_graph, arguments.output, 'trained_graph.pb', as_text=False)
        # tf.train.write_graph(trained_graph, arguments.output, 'trained_graph.pbtxt', as_text=True)


if __name__ == '__main__':
    main()
