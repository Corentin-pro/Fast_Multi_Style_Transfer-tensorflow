import argparse
import os

import cv2
import numpy as np

from src.functions import inverse_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--input-size', type=int, default=256, help='Shape of input to use (depends on checkpoint)')
    parser.add_argument('--inter', nargs='+', type=int, help='Interpolate between the 4 style given')
    arguments = parser.parse_args()

    style_control = []
    style_inter = arguments.inter
    if not style_inter:
        for style_index in range(16):
            style_control.append([0.0] * 16)
            style_control[-1][style_index] = 1
    else:
        for col in range(4):
            for row in range(4):
                style_index = (col % 4) + (row * 4)
                style_control.append([0.0] * 16)
                # top left style
                style_control[-1][style_inter[0]] = ((3 - row) / 3) * ((3 - col) / 3)
                # top right style
                style_control[-1][style_inter[1]] = (row / 3) * ((3 - col) / 3)
                # bottom left style
                style_control[-1][style_inter[2]] = ((3 - row) / 3) * (col / 3)
                # bottom right style
                style_control[-1][style_inter[3]] = (row / 3) * (col / 3)
    style_control = np.asarray(style_control, dtype=np.float)

    capture = cv2.VideoCapture(-1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    if tf.__version__.split('.')[0] == '2':
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
    import tensorflow.compat.v1 as tf1
    from engine_multi import EngineMultiStyle

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)
    with tf1.Session(config=session_config).as_default() as session:
        input_size = arguments.input_size
        engine = EngineMultiStyle(session, input_size, arguments.checkpoint_path)

        mosaic = np.zeros((4 * input_size, 4 * input_size, 3), dtype=np.uint8)
        while(True):
            # Capture frame-by-frame
            _, frame = capture.read()

            frame = cv2.resize(frame, (arguments.input_size, arguments.input_size))
            input_image = np.asarray(frame)

            outputs = engine.predict([input_image] * 16, style_control)
            for row in range(4):
                for col in range(4):
                    mosaic[
                        col * input_size:(col + 1) * input_size,
                        row * input_size:(row + 1) * input_size] = outputs[
                            (4 * col) + row]

            # Display the resulting frame
            cv2.imshow('original', input_image)
            cv2.imshow('style', mosaic)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
