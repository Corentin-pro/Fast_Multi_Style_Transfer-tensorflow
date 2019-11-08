import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image

from src.functions import inverse_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', help='Path to checkpoint to load')
    parser.add_argument('--style', nargs='+', default=None, help='List of weights for style')
    parser.add_argument('--images', help='Folder containing style images')
    parser.add_argument('--style-count', type=int, default=16, help='Number of style')
    parser.add_argument('--input-size', type=int, default=256, help='Shape of input to use (depends on checkpoint)')
    parser.add_argument('--output-size', type=int, default=None, help='Shape of output image')
    arguments = parser.parse_args()

    if arguments.style is None:
        style_control = None
        temp_style = [1.0] + ([0.0] * (arguments.style_count - 1))
        growing_style = 0
        previous_growing_style = -1
        style_rate = 0.02
    else:
        style_control = [float(value) for value in arguments.style]
        temp_style = None
        growing_style = None
        previous_growing_style = None
        style_rate = None

    if arguments.images:
        style_images = []
        paths = sorted(
            glob.glob(os.path.join(arguments.images, '*.jp*')),
            key=lambda x: int(os.path.basename(x)[:os.path.basename(x).find('-')]))
        for image_path in paths:
            style_images.append((
                os.path.basename(image_path),
                np.asarray(Image.open(image_path), dtype=np.uint8)[:, :, ::-1]))
    else:
        style_images= None

    capture = cv2.VideoCapture(-1)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    if tf.__version__.split('.')[0] == '2':
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
    import tensorflow.compat.v1 as tf1
    from engine import Engine

    gpu_options = tf1.GPUOptions(allow_growth=True)
    session_config = tf1.ConfigProto(gpu_options=gpu_options)

    with tf1.Session(config=session_config).as_default() as session:
        engine = Engine(session, arguments.input_size, arguments.checkpoint_path,
                        style_count=arguments.style_count, style_control=style_control, resize=arguments.output_size)

        auto_switch = True
        key_pressed = None
        while(True):
            # Capture frame-by-frame
            _, frame = capture.read()
            height, width, _ = frame.shape
            width_crop = (width - arguments.input_size) // 2
            height_crop = (height - arguments.input_size) // 2
            frame = frame[height_crop:-height_crop, width_crop:-width_crop, :]

            # frame = cv2.resize(frame, (arguments.input_size, arguments.input_size))
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
                    elif key_pressed == ord('p'):
                        previous_growing_style = growing_style
                        growing_style -= 1
                        if growing_style < 0:
                            growing_style = len(temp_style) - 1

                output = engine.predict([input_image], temp_style)[0]
                if style_images:
                    style_name, style_image = style_images[growing_style]
                    print(style_name, end='\r')
                    cv2.imshow('style', style_image)
                else:
                    print(('{:0.1f} ' * arguments.style_count).format(*temp_style), end='\r')
            else:
                output = engine.predict([input_image])[0]

            # Display the resulting frame
            cv2.imshow('original', input_image)
            cv2.imshow('output', output)
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
