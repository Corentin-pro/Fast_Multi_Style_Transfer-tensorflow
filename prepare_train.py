import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('style', help='Path to folder containing style files')
    parser.add_argument('--start', type=int, help='Start at specific index')
    parser.add_argument('--count', type=int, help='Number of style to train')
    parser.add_argument('--output', default='MST', help='Path to output folder')
    arguments = parser.parse_args()

    style_paths = sorted(os.listdir(arguments.style), key=lambda x: int(x.split('-')[0]))
    style_count = len(style_paths) if arguments.count is None else arguments.count
    style_index = 0
    if arguments.start:
        style_paths = style_paths[arguments.start:]
    with open('train_prepared.sh', 'w') as train_file:
        train_file.write('#!/usr/bin/env bash')
        for style_index, style_path in enumerate(style_paths[:style_count]):
            style_hot = ['1' if index == style_index else '0' for index in range(style_count)]
            train_file.write('\npython3 main.py -f 1 -gn 0 -p {} -n {} -b {} -tsd images/test -scw {} -sti {}'.format(
                arguments.output, 10 if style_index == 0 else 2,
                style_count, ' '.join(style_hot), os.path.join(arguments.style, style_path)))
            style_index += 1


if __name__ == '__main__':
    main()
