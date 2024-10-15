#!/usr/bin/env python3

import cv2
from cv2 import HoughCircles, HOUGH_GRADIENT, imread, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
import numpy as np
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

@cli.command()
@click.argument('path', type=Path)
def classify(path):
    from ast import literal_eval

    img = imread(path, cv2.IMREAD_COLOR)
    dot_colors = {}
    f = open('classify.json')
    for line in f.read().strip().strip('{}').strip().splitlines():
        if not line: continue
        name, values = line.rstrip(',').split(':')
        name, values = literal_eval(name.strip()), literal_eval(values.strip())
        dot_colors[name] = [tuple(x) for x in values]

    circles_by_color = {}
    blur = cv2.medianBlur(img, 7) # hmm, not sure what this does - but the internet told me it helps?
    for color, (lower, upper) in dot_colors.items():
        mask = cv2.inRange(blur, lower, upper)
        circles = HoughCircles(
            image=mask,
            method=HOUGH_GRADIENT,
            dp=1,
            # minDist=80,
            minDist=50,
            param1=20,
            param2=8,
        )
        circles_by_color[color] = np.round(circles).astype("int")

    for color, dots in circles_by_color.items():
        num = dots.shape[1] if dots is not None else 0
        print('{color.title()} Dots: {num}')


if __name__ == "__main__":
    cli()
