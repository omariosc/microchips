"""Generate a synthetic micromodel flooding sequence for demonstration.

The real micromodel imagery from the Journal of Molecular Liquids study is not
redistributed here. This script instead synthesises a ten-frame sequence with
the same visual characteristics the segmentation pipeline was tuned for:

  * a blobby, randomly-connected pore network of pale grey-blue solid grains,
  * pore space initially saturated with crude oil at the target colour #878874,
  * an invading brine front that displaces oil progressively over ten "hours",
    with viscous fingering so the front is irregular rather than flat.

Running `python demo/make_demo_data.py` writes `img/DEMO/T=1h.jpg` ... `T=10h.jpg`,
which `cla.py` and `vis.py` then consume unchanged.

Deterministic: seeded, so the same frames are produced on every run.
"""

import argparse
import os

import cv2
import numpy as np

OIL_BGR = np.array([0x74, 0x88, 0x87], dtype=np.float32)  # #878874 in BGR
SOLID_BGR = np.array([0xC9, 0xCD, 0xC8], dtype=np.float32)  # pale grey-blue grains
BRINE_BGR = np.array([0xE8, 0xEC, 0xE9], dtype=np.float32)  # displaced (water-filled) pore
SEED = 20250124  # the paper's acceptance date, for a stable arbitrary seed


def _smooth_noise(shape, scale, rng):
    """Band-limited noise: white noise blurred to a characteristic length scale."""
    noise = rng.standard_normal(shape).astype(np.float32)
    k = int(scale) | 1
    noise = cv2.GaussianBlur(noise, (k, k), 0)
    noise -= noise.min()
    return noise / (noise.max() + 1e-9)


def build_pore_network(height, width, rng, porosity=0.42):
    """Return a boolean mask that is True in pore space, False inside solid grains."""
    field = _smooth_noise((height, width), scale=45, rng=rng)
    # Threshold at the quantile that yields the requested porosity.
    pore = field < np.quantile(field, porosity)
    pore = pore.astype(np.uint8)
    # Open then close so grains have smooth, rounded outlines like sintered glass beads.
    ellipse = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    pore = cv2.morphologyEx(pore, cv2.MORPH_OPEN, ellipse(13))
    pore = cv2.morphologyEx(pore, cv2.MORPH_CLOSE, ellipse(17))
    return pore.astype(bool)


def render_frame(pore, oil, rng):
    """Composite a BGR frame from the pore mask and the current oil occupancy."""
    height, width = pore.shape
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[...] = SOLID_BGR
    frame[pore] = BRINE_BGR
    frame[oil] = OIL_BGR

    # Per-pixel shading so the phases are not perfectly flat colours.
    shade = _smooth_noise((height, width), scale=9, rng=rng)[..., None]
    frame *= 0.94 + 0.12 * shade

    # Dark outlines where solid meets fluid, as in the transmitted-light micrographs.
    edges = cv2.morphologyEx(
        pore.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    frame[edges] *= 0.45

    frame += rng.normal(0.0, 3.0, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


def main(out_dir, height, width, frames):
    rng = np.random.default_rng(SEED)
    os.makedirs(out_dir, exist_ok=True)

    pore = build_pore_network(height, width, rng)

    # Susceptibility to displacement: low values are swept first. A left-to-right
    # gradient sets the overall front direction; the noise term produces fingering.
    gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :].repeat(height, 0)
    susceptibility = 0.55 * gradient + 0.45 * _smooth_noise((height, width), 41, rng)

    # Swept fraction per frame: fast early breakthrough, then a plateau, which is the
    # characteristic shape of the nanofluid curves in Fig. 13 of the paper.
    swept = 1.0 - np.exp(-0.55 * np.arange(frames, dtype=np.float32))
    swept = swept / swept.max() * 0.82

    for i, fraction in enumerate(swept, start=1):
        cutoff = np.quantile(susceptibility[pore], fraction) if fraction > 0 else -1.0
        oil = pore & (susceptibility > cutoff)
        frame = render_frame(pore, oil, rng)
        path = os.path.join(out_dir, f"T={i}h.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{path}  oil occupies {oil.sum() / oil.size:.3f} of the frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="img/DEMO", help="output frame directory")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    main(args.out, args.height, args.width, args.frames)
