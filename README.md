# Filtered Back Projection from Scratch

A from-scratch Python implementation of filtered back projection (FBP), the
standard algorithm for reconstructing a CT image from its projections. Written
for learning: each step of the pipeline is its own small function.

The script simulates a parallel-beam scan of a circle, reconstructs it with
different numbers of projection angles, and shows the results side by side,
including an optional low-pass filtered reconstruction.

## Setup

Requires Python 3 with NumPy and Matplotlib.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install numpy matplotlib
```

On Windows, if PowerShell refuses to run the activate script, run this first
(applies to the current window only):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Usage

```bash
python fbp.py
```

To change the experiment, edit the `if __name__ == "__main__":` block at the
bottom of `fbp.py`:

| Setting | Meaning |
|---|---|
| `R, X, Y` | Circle radius and centre, in pixels (`X = Y = 0` is centred) |
| `mu` | Attenuation inside the circle, per pixel |
| `(4, 32, 128)` | Numbers of projection angles to compare |
| `sigma` | Detector noise, on the 0-255 intensity scale (0 = none) |
| `cutoff` | Low-pass cutoff in cycles/pixel (0.5 = Nyquist, i.e. no effect) |

## The pipeline

```
make_grid -> make_phantom                             1. Setup
make_sinogram (project -> to_intensity)               2. Simulate data
intensity_to_line_integral                            3. Pre-process
inverse_radon (make_filter -> filter_projection       4. Filtered back projection
               -> back_project)
show_results                                          5. Display
```

### 1. Setup
- **`make_grid`**: pixel coordinates for a 100 x 100 image and the positions
  of the 100 detector elements (-49.5 ... 49.5, one pixel apart).
- **`make_phantom`**: the true image, a circle with value `mu` inside and 0
  outside. Used only for display, not to make the data.

### 2. Simulate data
- **`project`**: exact line integrals through the circle at one angle,
  `mu x chord length`, where the chord is `2 * sqrt(R^2 - d^2)` for a ray at
  distance `d` from the centre.
- **`to_intensity`**: Beer-Lambert law, `I = 255 * exp(-p)`, plus optional
  Gaussian noise, clipped to 0-255. This is the "raw" detector reading.
- **`make_sinogram`**: repeats this for evenly spaced angles over 0-180
  degrees and stacks the results (angles x detector elements).

### 3. Pre-process
- **`intensity_to_line_integral`**: undoes Beer-Lambert, `p = -ln(I / 255)`.
  FBP works on line integrals, not raw intensities.

### 4. Filtered back projection
- **`make_filter`**: the ramp filter `|f|`, optionally multiplied by a
  smoothing window (`shepp-logan`, `cosine`, `hamming`, `hann`). The ramp is
  built in real space by default, which keeps the reconstructed values exact.
- **`make_lowpass`**: an optional Butterworth low-pass filter, multiplied
  onto the ramp. Reduces noise at the cost of sharpness.
- **`filter_projection`**: zero-pad, FFT, multiply by the filter, inverse FFT,
  remove the padding.
- **`back_project`**: smears each filtered projection back across the image
  along its rays, sums over all angles and scales by `pi / (number of angles)`.
- **`inverse_radon`**: runs the filtering and back projection together.

### 5. Display
- **`show_results`**: the phantom, the sinograms and the reconstructions side
  by side, a profile line through the circle, and (if given) an extra column
  with the low-pass filter and its reconstruction.

## What to look for

- **Sinogram**: an off-centre point traces a sine wave as the angle changes,
  hence the name.
- **Few angles**: streaks across the image. These come from missing views, so
  only more angles really remove them.
- **Many angles**: the circle comes back at the correct value `mu`, with only
  noise and slightly soft edges.
- **Low-pass filter**: less noise, blurrier edges, and some ringing at the
  edge if the cutoff is low or the roll-off is steep (high `order`).

## Conventions

- Units are pixels; the detector spacing equals the pixel size.
- `y` points up in the image, so row 0 is the top.
- Angles are in radians inside the code, shown in degrees in the plots.
- Parallel-beam geometry: angles cover 0-180 degrees, since 180-360 repeats them.

## Reference

A. C. Kak and M. Slaney, *Principles of Computerized Tomographic Imaging*,
SIAM, 2001. Chapter 3 covers FBP and the real-space ramp filter used here.
