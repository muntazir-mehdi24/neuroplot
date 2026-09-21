# Changelog

## [0.1.3] - 2026-09-21
- **Fixed:** Desktop live-window rendering by enforcing explicit `plt.show(block=False)` and robust GUI backend fallback (`TkAgg`/`QtAgg`).
- **Improved:** Backward compatibility layer for legacy scalar float loss inputs (`viz.step(epoch, loss=val)`).
- **Documentation:** Complete, production-grade rewrite of `README.md` detailing all API specifications, hooks, and architectures.

## [0.1.2] - 2026-09-20
- **Added:** Standalone interactive window rendering alongside background GIF recording.
- **Improved:** Robust backend detection ensuring independent control of `save_gif` and desktop GUI windows.

## [0.1.1] - 2026-09-18
- Initial stable release of `neuroplot` with dynamic grid plotting, EMA smoothing, and checkpoint saver.