import argparse
import os

import cv2
import numpy as np

from src.functions import inverse_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--style', nargs='+',
                        default=[1.] * 16, help='List of weights for style')
    parser.add_argument('--input_size', type=int, default=256, help='Shape of input to use (depends on checkpoint)')
    arguments = parser.parse_args()

    style_control = [float(value) for value in arguments.style]

    capture = cv2.VideoCapture(-1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    from engine import Engine

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)

    with tf1.Session(config=session_config).as_default() as session:
        engine = Engine(session, style_control, arguments.input_size, arguments.checkpoint_path)

        while(True):
            # Capture frame-by-frame
            _, frame = capture.read()

            frame = cv2.resize(frame, (arguments.input_size, arguments.input_size))
            input_image = np.asarray(frame)

            output = engine.predict([input_image])[0]
            output[output > 255] = 255
            output[output < 0] = 0

            # Display the resulting frame
            cv2.imshow('original', input_image)
            cv2.imshow('style', output.astype(np.uint8))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
