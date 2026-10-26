# LMM40X-P01 Gaussian-beam focusing simulation

`lmm40x.py` contains the model and `run_sim.py` runs the sweep. Output goes to `results/`:
`spot_*.png` per case, `summary_*.png`, `summary.csv`/`.md`, `layout.png`, `pupil.png`.

Run it with `.venv/Scripts/python.exe run_sim.py` (about 1 minute).

## Objective (from the Thorlabs Zemax file `265_LMM40X-P01.zmx`)
- This is an infinity-corrected Schwarzschild objective. The collimated beam passes through the hole in
  the concave primary (R = 16.18 mm) and hits the small convex secondary (R = 6.175 mm, clear radius 2.6 mm).
  The secondary sends it back onto the primary annulus (4 to 10.75 mm), which focuses it past the secondary.
- From the real ray trace: EFL is 5.03 mm, NA is 0.507 and WD is 7.76 mm.
  The RMS wavefront error is 4.3 nm (λ/150 at 633 nm), so the objective is diffraction limited at both wavelengths.
  The sine condition holds to better than 1e-4.
- The central obscuration is set by the secondary housing (S7, r < 2.5 mm). It blocks every entrance-pupil
  radius below **1.22 mm** (ε = 0.479, 23% of the pupil area).
  The three curved spiders block another ≈5%, and the open area is 71.7% of the 5.1 mm pupil.

## Method
1. Exact real-ray trace through both spherical mirrors and all obscurations, including the UDA spider polygons.
2. Debye–Wolf (Richards–Wolf) **vector** diffraction integral over the traced output directions.
   - The apodisation is √(dΩ/dA), taken from the numerical Jacobian of the trace.
   - The phase includes the real optical path difference.
   - Polarisation is rotated with each ray's s/p basis. Mirrors are treated as ideal, with no coating retardance.
3. Validation: the scalar result matches the analytic annular-Airy pattern to 0.3%.

## Why the poppy notebook is not right
- **No central obscuration** (`obscuration_radius = 0`). In reality the inner 1.22 mm of the pupil is dark,
  which dominates both the spot shape and the throughput.
- **Wrong spiders.** It uses straight vanes 0.24 mm wide in the pupil. The real spiders are curved,
  0.6 mm wide, and sit in the converging beam.
- **Paraxial scalar Fraunhofer model.** At NA 0.5 this misses the vector elongation along the
  polarisation direction (about 10%) and the Ez component (about 7%).

## Key results (x-polarised input; D = 1/e² input diameter)
- **Throughput.** It peaks at about 46% for D ≈ 5.2 mm. Below that the obscuration clips the beam; above it the
  5.1 mm pupil does. A beam with **D ≤ 2 mm is almost entirely blocked** (1 mm: 6e-6, 2 mm: 4.6%).
- **Spot shape.** For D ≳ 4 mm the spot is the annular-pupil (ε ≈ 0.48) Airy pattern. About 51% of the power is
  in the core and the first ring is about 9% of the peak. The phase is flat across the core and jumps by π at each ring.
  - Smaller beams do not give larger Gaussian spots, as w0 = λf/πw would suggest.
  - They illuminate only a thin annulus, which gives a Bessel-like spot with stronger rings.
- **Best-fit Gaussian at best focus, D = 5 mm:**

| λ | LSQ intensity fit wx × wy (1/e²) | mode-matched (max overlap) wx × wy | max coupling η | FWHM x × y |
|---|---|---|---|---|
| 632.8 nm | 0.519 × 0.472 µm | 0.394 × 0.382 µm | 45% | 0.636 × 0.590 µm |
| 2923 nm  | 2.40 × 2.18 µm | 1.82 × 1.77 µm | 45% | 2.94 × 2.73 µm |

- **Which fit to use.** "LSQ" fits the intensity profile of the core. "Mode-matched" is the TEM00 waist that
  captures the most power from the spot. At most 45% of the focused light can couple into a single Gaussian mode,
  so overall input-to-Gaussian efficiency is T·η ≈ 21%.
- **Scaling.** All widths scale with λ. A useful rule is w_LSQ ≈ 0.82 λ (along the polarisation) × 0.75 λ (across it).

See `results/summary.md` for every diameter.
