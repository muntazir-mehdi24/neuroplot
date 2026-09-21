# Changelog

## [0.1.4] - 2026-09-21
- **Fixed:** Prevented importing `neuroplot` from mutating Matplotlib's global backend and added headless fallback handling when GUI backends cannot be initialized.
- **Fixed:** Added validation for required models, data, boundary input shapes, and latent-space layers, raising clear `ValueError` messages for invalid configurations.
- **Fixed:** Corrected grid-layout axis trimming so the visualizer exposes exactly one axis per requested plot.
- **Fixed:** Ensured custom plot hooks execute and excluded diagnostic/custom plots from ordinary metric history tracking.
- **Fixed:** Scoped interactive show/pause behavior to each visualizer's own figure canvas, avoiding non-interactive backend warnings and interference from unrelated figures.
- **Tests:** Verified the fixes with the complete test suite (`4 passed`).

## [0.1.3] - 2026-09-21
- **Fixed:** Desktop live-window rendering by enforcing explicit `plt.show(block=False)` and robust GUI backend fallback (`TkAgg`/`QtAgg`).
- **Improved:** Backward compatibility layer for legacy scalar float loss inputs (`viz.step(epoch, loss=val)`).
- **Documentation:** Complete, production-grade rewrite of `README.md` detailing all API specifications, hooks, and architectures.

## [0.1.2] - 2026-09-20
- **Added:** Standalone interactive window rendering alongside background GIF recording.
- **Improved:** Robust backend detection ensuring independent control of `save_gif` and desktop GUI windows.

## [0.1.1] - 2026-09-18
- Initial stable release of `neuroplot` with dynamic grid plotting, EMA smoothing, and checkpoint saver.