#!/usr/bin/env python3

from cv2 import HoughCircles, HOUGH_GRADIENT, imread, IMREAD_GRAYSCALE
from pathlib import Path
import click

@click.group()
def cli(): pass

@cli.command()
@click.argument("path", type=Path)
def find(path):
    """Count the number of dots in an image.

    Supports all image file formats that open-cv2 does.

    Parameters
    ----------
    path: pathlib.Path
        A relative or absolute path to an image file.

    Returns
    -------
    None
        Output is written to stdout since this is a cli tool.
    """

    img = imread(path, IMREAD_GRAYSCALE)
    dots = HoughCircles(img, HOUGH_GRADIENT, dp=2, minDist=1)

    if dots is None:
        print("There are no dots.")
        return

    num = dots.shape[1]
    if num == 1:
        print("There is one dot.")
    else:
        print(f"There are {num} dots.")


if __name__ == "__main__":
    cli()
