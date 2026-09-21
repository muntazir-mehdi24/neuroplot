# Changelog

## [0.1.3] - 2026-09-20
- **Added:** Full standalone support for interactive display window rendering (`plt.show(block=False)`) alongside background GIF recording.
- **Added:** Robust backend detection ensuring independent control of `save_gif` and interactive desktop GUI windows.
- **Improved:** Backward compatibility layer for legacy scalar float loss inputs (`viz.step(epoch, loss=val)`).
- **Added:** Verified telemetry features for Gradient Norms (`grad_norm`), Regression Fit tracking (`regression_fit`), and Custom Hook functions.

## [0.1.1] - 2026-09-18
- Initial stable release of `neuroplot` with dynamic grid plotting, EMA smoothing, and checkpoint saver.