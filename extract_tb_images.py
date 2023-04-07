"""
Usage:
python extract_tb_images.py --path output/cfgs/kitti_models/pv_rcnn_ssl_60/test/tensorboard/ \
--outdir output/cfgs/kitti_models/pv_rcnn_ssl_60/test/tensorboard
"""
import argparse
import pathlib

import cv2
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def extract_images_from_tb():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="experiment path")
    parser.add_argument('--outdir', type=str, required=True, help="will be used to store images")
    args = parser.parse_args()

    event_acc = event_accumulator.EventAccumulator(
        args.path, size_guidance={'images': 0})
    event_acc.Reload()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for tag in event_acc.Tags()['images']:
        events = event_acc.Images(tag)

        tag_name = tag.replace('/', '_')
        dirpath = outdir / tag_name
        dirpath.mkdir(exist_ok=True, parents=True)

        for index, event in enumerate(events):
            image_buffer = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            outpath = dirpath / '{:04}.jpg'.format(index)
            cv2.imwrite(outpath.as_posix(), cv2.imdecode(image_buffer, cv2.IMREAD_COLOR))


if __name__ == '__main__':
    extract_images_from_tb()