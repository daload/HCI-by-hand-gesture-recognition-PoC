import argparse
import logging
import os

from detector.ssd_mobilenetv3 import SSDMobilenet

from constants import targets, threshold

from v1 import V1
from v2 import V2
from v3 import V3


def parse_arguments(params=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v",
        "--version",
        required=True,
        choices=range(1, 4),
        type=int,
        help="Version to run"
    )

    known_args, _ = parser.parse_known_args(params)
    return known_args


def _load_model():
    """
    Load the CNN model
    """
    model_path = 'SSDLite.pth'
    ssd_mobilenet = SSDMobilenet(num_classes=len(targets) + 1)
    if not os.path.exists('SSDLite.pth'):
        logging.info(f"Model not found {model_path}")
        raise FileNotFoundError

    ssd_mobilenet.load_state_dict(model_path, map_location='cpu')
    ssd_mobilenet.eval()
    return ssd_mobilenet


if __name__ == '__main__':
    args = parse_arguments()
    model = _load_model()
    if args.version == 1:
        V1.run(model, threshold=threshold)
    if args.version == 2:
        V2.run(model, threshold=threshold)
    if args.version == 3:
        V3.run(model, threshold=threshold)
