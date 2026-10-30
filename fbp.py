import numpy as np
import matplotlib.pyplot as plt

 
# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------

def make_grid(n=100, delta=1.0):
    c = (np.arange(n) - (n - 1) / 2) * delta     # -49.5 ... 49.5
    xx, yy = np.meshgrid(c, -c)                  # y flipped so +y is at the top
    s = c.copy()                                 # detector element centres
    return xx, yy, s

def make_phantom(xx, yy, R=25.0, X=0.0, Y=0.0, mu=0.02):
    inside = (xx - X) ** 2 + (yy - Y) ** 2 <= R ** 2
    return np.where(inside, mu, 0.0)
 
 
# ---------------------------------------------------------------------------
# 2. Simulate data
# ---------------------------------------------------------------------------
 
def project(theta, s, R=25.0, X=0.0, Y=0.0, mu=0.02):
    # A. The circle's centre lands at the s of the ray through (X, Y)
    s_centre = X * np.cos(theta) + Y * np.sin(theta)
 
    # B. Signed distance from each ray to the centre, along the detector axis
    d = s - s_centre
 
    # C. A ray hits the circle if it passes closer than the radius
    hits = np.abs(d) < R
 
    # D. Pythagoras: half-chord^2 + d^2 = R^2, so chord = 2*sqrt(R^2 - d^2)
    chord = np.zeros_like(s, dtype=float)
    chord[hits] = 2 * np.sqrt(R**2 - d[hits]**2)
    
    # E. f is constant (mu) along the chord and 0 elsewhere,
    #    so the line integral is mu * chord length
    p = mu * chord
 
    return p

def to_intensity(p, I0=255.0, sigma=0.5, rng=None):
    rng = np.random.default_rng(rng)
    I = I0 * np.exp(-np.asarray(p, dtype=float))   # Beer-Lambert
    if sigma > 0:
        I = I + rng.normal(0.0, sigma, size=I.shape)
    return np.clip(I, 0.0, I0)

def make_sinogram(n_angles, s, R=25.0, X=0.0, Y=0.0, mu=0.02,
                  I0=255.0, sigma=0.5, rng=None):
    rng = np.random.default_rng(rng)    # one generator shared by all angles
    thetas = np.arange(n_angles) * np.pi / n_angles     # theta_m = m * pi / M
    sino = np.zeros((n_angles, len(s)))
    for m, theta in enumerate(thetas):
        p = project(theta, s, R=R, X=X, Y=Y, mu=mu)
        sino[m] = to_intensity(p, I0=I0, sigma=sigma, rng=rng)
    return sino, thetas


# ---------------------------------------------------------------------------
# 3. Pre-process
# ---------------------------------------------------------------------------

def intensity_to_line_integral(I, I0=255.0, eps=1e-6):
    I = np.maximum(np.asarray(I, dtype=float), eps)
    return -np.log(I / I0)

# ---------------------------------------------------------------------------
# 4. Filtered back projection
# ---------------------------------------------------------------------------

def padded_length(n):
    return int(2 ** np.ceil(np.log2(2 * n)))      # 100 -> 256

def make_filter(L, name="ramp", delta=1.0, ramp="spatial"):
    f = np.fft.fftfreq(L, d=delta)
    f_max = 1.0 / (2.0 * delta)

    # the ramp
    if ramp == "frequency":
        H = np.abs(f)
    elif ramp == "spatial":
        n = np.fft.fftfreq(L, d=1.0 / L).astype(int)
        h = np.zeros(L)
        h[0] = 1.0 / (4.0 * delta**2)
        odd = (n % 2) == 1
        h[odd] = -1.0 / (np.pi**2 * n[odd].astype(float)**2 * delta**2)
        H = delta * np.real(np.fft.fft(h))

    # the window
    if   name == "ramp":        W = np.ones(L)
    elif name == "shepp-logan": W = np.sinc(f / (2 * f_max))
    elif name == "cosine":      W = np.cos(np.pi * f / (2 * f_max))
    elif name == "hamming":     W = 0.54 + 0.46 * np.cos(np.pi * f / f_max)
    elif name == "hann":        W = 0.5 * (1 + np.cos(np.pi * f / f_max))

    return H * W

def make_lowpass(L, cutoff=0.25, order=4, delta=1.0):
    # Butterworth low-pass: 1 below the cutoff, rolls off to 0 above it.
    # cutoff in cycles/pixel (Nyquist = 0.5); higher order = sharper roll-off.
    f = np.fft.fftfreq(L, d=delta)
    return 1.0 / (1.0 + (np.abs(f) / cutoff) ** (2 * order))

def filter_projection(p, H):
    p = np.asarray(p, dtype=float)
    n = p.size
    L = H.size

    p_pad = np.zeros(L)          # 1. zero-pad to length L
    p_pad[:n] = p
    P = np.fft.fft(p_pad)        # 2. FFT
    Q = P * H                    # 3. multiply by the filter
    q_pad = np.real(np.fft.ifft(Q))   # 4. inverse FFT, real part
    return q_pad[:n]             # 5. remove the padding

def back_project(Q, thetas, xx, yy, s):
    Q = np.atleast_2d(Q)
    n_det = s.size
    delta = s[1] - s[0]
    img = np.zeros_like(xx, dtype=float)

    for q, theta in zip(Q, thetas):
        s_pix = xx * np.cos(theta) + yy * np.sin(theta)    # where each pixel lands

        u = (s_pix - s[0]) / delta                          # fractional index
        inside = (u >= 0) & (u <= n_det - 1)
        k0 = np.clip(np.floor(u).astype(int), 0, n_det - 2)
        w = u - k0
        val = (1 - w) * q[k0] + w * q[k0 + 1]               # linear interpolation

        img += np.where(inside, val, 0.0)                   # accumulate

    return img * np.pi / len(thetas)                        # scale

def inverse_radon(P, thetas, xx, yy, s, filter_name="ramp", ramp="spatial",
                  lowpass=None, return_filtered=False):
    P = np.atleast_2d(np.asarray(P, dtype=float))

    L = padded_length(P.shape[1])
    H = make_filter(L, name=filter_name, delta=s[1] - s[0], ramp=ramp)
    if lowpass is not None:                                # low-pass after the ramp
        H = H * make_lowpass(L, cutoff=lowpass, delta=s[1] - s[0])

    Q = np.array([filter_projection(p, H) for p in P])    # 1. filter every projection
    img = back_project(Q, thetas, xx, yy, s)               # 2. back-project

    return (img, Q) if return_filtered else img


# ---------------------------------------------------------------------------
# 5. Display
# ---------------------------------------------------------------------------

def show_results(phantom, results, xx, yy, s, Y=0.0, mu=None, I0=255.0,
                 lowpass=None):
    """Phantom, sinograms and reconstructions side by side.

    Layout (one column per number of angles):

        phantom          | sinogram M1      | sinogram M2      | ...
        profile through  | reconstruction   | reconstruction   | ...
        the circle       | M1               | M2               |

    results : list of dicts, one per number of angles, each with keys
        "M"      number of angles
        "sino"   (M, n) intensity sinogram (from make_sinogram)
        "thetas" (M,) angles in radians
        "img"    (n, n) reconstruction (from inverse_radon)
    lowpass : optional dict, adds a final column
        "M"      number of angles
        "img"    reconstruction with the low-pass applied
        "cutoff" low-pass cutoff (cycles/pixel)
    """

    if mu is None:
        mu = phantom.max()
    n_cols = 1 + len(results) + (lowpass is not None)
    half = (xx.max() - xx.min()) / 2 + 0.5
    extent_img = [-half, half, -half, half]          # x and y in pixels
    ds = s[1] - s[0]

    # One grey scale for the phantom and every reconstruction
    vmin, vmax = -0.25 * mu, 1.25 * mu

    ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"]

    fig, axes = plt.subplots(2, n_cols, figsize=(3.3 * n_cols, 6.9),
                             constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")

    def style(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink2)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=ink2, labelsize=8, width=0.8)
        ax.title.set_color(ink)

    # --- top-left: the phantom ---
    ax = axes[0, 0]
    im_recon = ax.imshow(phantom, cmap="gray", vmin=vmin, vmax=vmax, extent=extent_img)
    ax.axhline(Y, color=series[0], lw=1, ls=":", alpha=0.9)
    ax.set_title("True phantom", fontsize=10)
    ax.set_xlabel("x (pixels)", fontsize=8, color=ink2)
    ax.set_ylabel("y (pixels)", fontsize=8, color=ink2)
    style(ax)

    # --- bottom-left: profiles along y = Y ---
    ax = axes[1, 0]
    row = np.argmin(np.abs(yy[:, 0] - Y))
    x_line = xx[row]
    ax.plot(x_line, phantom[row], color=ink, lw=2, label="true", drawstyle="steps-mid")
    for k, res in enumerate(results):
        ax.plot(x_line, res["img"][row], color=series[k % len(series)], lw=1.5,
                label=f"{res['M']} angles")
    if lowpass is not None:
        ax.plot(x_line, lowpass["img"][row], color=series[len(results) % len(series)],
                lw=1.5, label=f"{lowpass['M']} angles + low-pass")
    ax.axhline(0, color=grid_c, lw=0.8, zorder=0)
    ax.set_ylim(vmin, vmax)
    ax.set_xlim(x_line[0], x_line[-1])
    ax.set_title(f"Profile along y = {yy[row, 0]:g}", fontsize=10)
    ax.set_xlabel("x (pixels)", fontsize=8, color=ink2)
    ax.set_ylabel("attenuation", fontsize=8, color=ink2)
    ax.legend(fontsize=7, frameon=False, labelcolor=ink2, loc="upper left")
    style(ax)

    # --- one column per number of angles ---
    for k, res in enumerate(results, start=1):
        M, sino, img = res["M"], res["sino"], res["img"]
        deg = np.rad2deg(res["thetas"])
        dtheta = 180.0 / M

        # sinogram: rows = angles, columns = detector position
        ax = axes[0, k]
        im_sino = ax.imshow(sino, cmap="gray", vmin=0, vmax=I0, aspect="auto",
                            interpolation="nearest",
                            extent=[s[0] - ds / 2, s[-1] + ds / 2,
                                    deg[-1] + dtheta / 2, deg[0] - dtheta / 2])
        ax.set_title(f"Sinogram, {M} angles", fontsize=10)
        ax.set_xlabel("detector position s", fontsize=8, color=ink2)
        ax.set_ylabel("angle (degrees)", fontsize=8, color=ink2)
        style(ax)

        # reconstruction
        ax = axes[1, k]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, extent=extent_img)
        ax.set_title(f"FBP, {M} angles", fontsize=10)
        ax.set_xlabel("x (pixels)", fontsize=8, color=ink2)
        style(ax)

    # --- extra column: low-pass ---
    if lowpass is not None:
        k = len(results) + 1
        L = padded_length(len(s))
        f = np.fft.fftfreq(L, d=ds)
        H_ramp = make_filter(L, delta=ds)
        H_lp = H_ramp * make_lowpass(L, cutoff=lowpass["cutoff"], delta=ds)
        pos = f >= 0

        # top: filter responses
        ax = axes[0, k]
        ax.plot(f[pos], H_ramp[pos], color=ink, lw=2, label="ramp")
        ax.plot(f[pos], H_lp[pos], color=series[len(results) % len(series)], lw=2,
                label=f"ramp x low-pass")
        ax.axvline(lowpass["cutoff"], color=ink2, lw=0.8, ls=":")
        ax.set_xlim(0, f[pos].max())
        ax.set_ylim(0, None)
        ax.set_title(f"Filter, cutoff {lowpass['cutoff']:g}", fontsize=10)
        ax.set_xlabel("frequency (cycles/pixel)", fontsize=8, color=ink2)
        ax.set_ylabel("H(f)", fontsize=8, color=ink2)
        ax.legend(fontsize=7, frameon=False, labelcolor=ink2, loc="upper left")
        style(ax)

        # bottom: low-pass reconstruction
        ax = axes[1, k]
        ax.imshow(lowpass["img"], cmap="gray", vmin=vmin, vmax=vmax, extent=extent_img)
        ax.set_title(f"FBP + low-pass, {lowpass['M']} angles", fontsize=10)
        ax.set_xlabel("x (pixels)", fontsize=8, color=ink2)
        style(ax)

    cb = fig.colorbar(im_sino, ax=axes[0, 1:len(results) + 1], shrink=0.85, pad=0.01)
    cb.set_label("intensity I", fontsize=8, color=ink2)
    cb.ax.tick_params(labelsize=7, colors=ink2)
    cb = fig.colorbar(im_recon, ax=axes[1, 1:], shrink=0.85, pad=0.01)
    cb.set_label("attenuation", fontsize=8, color=ink2)
    cb.ax.tick_params(labelsize=7, colors=ink2)

    plt.show()

    return fig


if __name__ == "__main__":
    R, X, Y, mu = 25, 20, 10, 0.02

    xx, yy, s = make_grid(100)
    phantom = make_phantom(xx, yy, R, X, Y, mu)

    results = []
    for M in (4, 32, 128):
        sino, thetas = make_sinogram(M, s, R, X, Y, mu, sigma=0.5)
        P = intensity_to_line_integral(sino)
        img = inverse_radon(P, thetas, xx, yy, s)
        results.append(dict(M=M, sino=sino, thetas=thetas, img=img))

    # Low-pass version of the last reconstruction
    cutoff = 0.6                     # cycles/pixel (Nyquist = 0.5)
    img_lp = inverse_radon(P, thetas, xx, yy, s, lowpass=cutoff)
    lowpass = dict(M=M, img=img_lp, cutoff=cutoff)

    show_results(phantom, results, xx, yy, s, Y=Y, mu=mu, lowpass=lowpass)