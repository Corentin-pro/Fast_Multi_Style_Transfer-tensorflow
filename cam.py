import argparse
import os

import cv2
import numpy as np

from src.functions import inverse_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--style', nargs='+', default=None, help='List of weights for style')
    parser.add_argument('--input-size', type=int, default=256, help='Shape of input to use (depends on checkpoint)')
    parser.add_argument('--output-size', type=int, default=None, help='Shape of output image')
    arguments = parser.parse_args()

    if arguments.style is None:
        style_control = None
        temp_style = [1.0] + ([0.0] * 15)
        growing_style = 0
        previous_growing_style = -1
        style_rate = 0.02
    else:
        style_control = [float(value) for value in arguments.style]
        temp_style = None
        growing_style = None
        previous_growing_style = None
        style_rate = None

    capture = cv2.VideoCapture(-1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import tensorflow.compat.v1 as tf1
    from engine import Engine

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)

    with tf1.Session(config=session_config).as_default() as session:
        engine = Engine(session, arguments.input_size, arguments.checkpoint_path,
                        style_control=style_control, resize=arguments.output_size)

        auto_switch = True
        key_pressed = None
        while(True):
            # Capture frame-by-frame
            _, frame = capture.read()

            frame = cv2.resize(frame, (arguments.input_size, arguments.input_size))
            input_image = np.asarray(frame)

            if style_control is None:
                if auto_switch:
                    if temp_style[previous_growing_style] > 0:
                        temp_style[previous_growing_style] -= style_rate
                        if temp_style[previous_growing_style] < 0:
                            temp_style[previous_growing_style] = 0
                    if temp_style[growing_style] < 3:
                        temp_style[growing_style] += style_rate
                    else:
                        temp_style[previous_growing_style] = 0
                        temp_style[growing_style] = 1
                        previous_growing_style = growing_style
                        growing_style = (growing_style + 1) % (len(temp_style) - 8)
                else:
                    temp_style[previous_growing_style] = 0
                    temp_style[growing_style] = 1
                    if key_pressed == ord('n'):
                        previous_growing_style = growing_style
                        growing_style = (growing_style + 1) % len(temp_style)

                output = engine.predict([input_image], temp_style)[0]
                print(('{:0.2f} ' * 16).format(*temp_style), end='\r')
            else:
                output = engine.predict([input_image])[0]

            # Display the resulting frame
            cv2.imshow('original', input_image)
            cv2.imshow('style', output)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                break
            elif key_pressed == ord('a'):
                auto_switch = not auto_switch

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
