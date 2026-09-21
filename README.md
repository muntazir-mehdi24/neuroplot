# NeuroPlot 🚀

[![PyPI version](https://img.shields.io/pypi/v/neuroplot.svg)](https://pypi.org/project/neuroplot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

**NeuroPlot** is a lightweight, zero-boilerplate real-time telemetry and visualization library built specifically for PyTorch. It bridges the gap between bulky web dashboards (like TensorBoard or WandB) and messy terminal printouts. It enables machine learning practitioners to monitor training dynamics, gradient health, decision boundaries, latent representations, and custom metrics live on their desktop while optionally compiling training progressions into animated GIFs.

---

## 🏗️ Core Architecture & Design Philosophy

NeuroPlot is designed around three core principles:
1. **Low Overhead:** By decoupling display updates via `update_every` steps and utilizing non-blocking canvas event flushing (`plt.show(block=False)`), visualization overhead has a negligible impact on epoch training time.
2. **Decoupled Environment Intelligence:** Environment rendering adapts automatically. Local machines spawn native interactive GUI windows (`TkAgg`/`QtAgg`), while headless cloud instances or CI/CD pipelines seamlessly degrade to the non-interactive `Agg` backend without throwing display errors.
3. **Flexible Telemetry Ingestion:** Supports both modern dictionary-based multi-metric loss tracking and legacy scalar float inputs for seamless drop-in backward compatibility.

---

## 🌟 Key Features Breakdown

* **Live Interactive Telemetry:** Real-time multi-metric plotting with dual-line tracking (Raw values in transparent gray, Exponential Moving Average trendlines in bold color).
* **Backward Compatible API:** Native support for both modern multi-metric dictionaries (`loss={"train": val1, "val": val2}`) and legacy single scalar floats (`viz.step(epoch, loss=val)`).
* **Advanced PyTorch Diagnostics:**
  * `"grad_norm"`: Monitors total parameter gradient norms to detect exploding or vanishing gradients.
  * `"regression_fit"`: Scatter plots actual targets versus model predictions against an ideal $y=x$ reference line.
  * `"boundary"`: Visualizes 2D classification decision boundaries updated dynamically every step.
  * `"latent"`: Projects internal linear layer activations into a 2D scatter plot to inspect embedding spaces.
* **Custom Plot Hooks:** Inject arbitrary user-defined subplots (such as weight distributions, confusion matrices, or residual histograms) using simple callback signatures.
* **Automatic Headless Fallback:** Safely detects missing display drivers and switches to file-backed rendering (`Agg`).
* **GIF Export & Model Checkpointing:** Automatically records training iterations into an animated GIF via `imageio` and serializes best-performing model weights based on loss or accuracy benchmarks.

---

## 📦 Installation

Install the stable release from PyPI:

    pip install neuroplot

Or install in editable mode for local development from source:

    git clone https://github.com/muntazir-mehdi24/neuroplot.git
    cd neuroplot-repo
    pip install -e .

---

## 🚀 Quick Start Guide

A minimal example demonstrating model telemetry, scalar loss tracking, live window rendering, and clean teardown:

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from neuroplot import LiveVisualizer

    # 1. Define model, optimizer, and loss
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 2. Initialize the visualizer
    viz = LiveVisualizer(
        plots=["loss", "grad_norm"],
        model=model,
        update_every=1
    )

    # 3. Training Loop
    X_dummy = torch.randn(32, 2)
    y_dummy = torch.randn(32, 1)

    for epoch in range(1, 51):
        optimizer.zero_grad()
        preds = model(X_dummy)
        loss = criterion(preds, y_dummy)
        loss.backward()
        optimizer.step()
        
        # Legacy scalar float backward-compatible logging
        viz.step(epoch=epoch, loss=loss.item())

    viz.close()
    plt.show() # Keeps window active post-training

---

## 🔬 Advanced Usage: Multi-Metric Dictionaries & Custom Hooks

This comprehensive example demonstrates multi-metric dictionary loss tracking, model checkpointing, GIF generation, and a custom residual distribution hook.

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from neuroplot import LiveVisualizer

    # Synthetic Regression Data
    torch.manual_seed(42)
    X = torch.randn(200, 2)
    y = X[:, 0:1] + 2 * X[:, 1:2] + 0.1 * torch.randn(200, 1)

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        def forward(self, x):
            return self.net(x)

    model = Regressor()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # User-defined custom hook for residual histogram analysis
    def custom_residual_plot(ax, model, data):
        X_val, y_val = data
        model.eval()
        with torch.no_grad():
            preds = model(X_val)
            residuals = (y_val - preds).cpu().numpy()
        model.train()
        ax.hist(residuals, bins=15, color='orange', edgecolor='black', alpha=0.7)
        ax.set_title("Custom Hook: Residuals Distribution")
        ax.grid(True, linestyle='--', alpha=0.6)

    # Initialize LiveVisualizer with advanced features
    viz = LiveVisualizer(
        plots=["loss", "grad_norm", "regression_fit", "custom"],
        model=model,
        data=(X, y),
        update_every=5,
        save_gif=True,
        gif_name="advanced_training.gif",
        save_best_model=True,
        checkpoint_path="best_regressor.pth",
        smooth_alpha=0.1,
        custom_plot_fn=custom_residual_plot
    )

    print("Starting training telemetry session...")
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        preds = model(X)
        train_loss = criterion(preds, y)
        train_loss.backward()
        optimizer.step()
        
        # Modern multi-metric dictionary tracking
        val_loss = train_loss.item() * 1.15
        viz.step(epoch=epoch, loss={"train": train_loss.item(), "val": val_loss})

    viz.close()
    plt.show()
    print("Training session completed successfully!")

---

## 📋 API Reference & Argument Specifications

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `plots` | `List[str]` | **Required** | List of plots to render. Valid options: `"loss"`, `"accuracy"`, `"grad_norm"`, `"regression_fit"`, `"boundary"`, `"latent"`, `"custom"`. |
| `model` | `nn.Module` | `None` | PyTorch model reference required for gradient norm calculations, weights, and layer hook extractions. |
| `data` | `Tuple[Tensor, Tensor]` | `None` | Validation or evaluation dataset tuple `(X, y)` used by decision boundaries and regression plots. |
| `update_every` | `int` | `50` | Epoch frequency interval between screen redraws and GIF frame captures. |
| `save_gif` | `bool` | `False` | Enables frame capture and animated GIF compilation. Requires the `imageio` package. |
| `gif_name` | `str` | `"training_progress.gif"` | Output filename for the exported training progression animation. |
| `save_best_model` | `bool` | `False` | Automatically serializes model state dict when a new minimum loss or maximum accuracy is reached. |
| `checkpoint_path` | `str` | `"best_model.pth"` | Target file path for saving best model checkpoint weights. |
| `smooth_alpha` | `float` | `0.1` | Exponential Moving Average (EMA) smoothing constant ($0 < \alpha \le 1$). Lower values produce smoother trend curves. |
| `custom_plot_fn` | `Callable` | `None` | User callback function with signature `fn(ax, model, data)` mapped directly to the `"custom"` plot slot. |

---

## ❓ Troubleshooting & FAQs

* **Q: The script finishes running, but no interactive window appears or stays open.**
  * *A:* Ensure you call `viz.close()` followed by an explicit `plt.show()` at the very end of your training script. Without a blocking call, Python terminates immediately upon loop completion, destroying GUI windows.
* **Q: I get a `FigureCanvasAgg is non-interactive` warning.**
  * *A:* This occurs if Matplotlib defaults to the non-interactive backend. NeuroPlot automatically tries to enforce `TkAgg`/`QtAgg` on desktop environments, but ensure you have a standard GUI framework installed (`pip install tkinter` or PyQt).

---

## 🤝 Contributing
Contributions, bug reports, and pull requests are warmly welcomed! Please visit the [GitHub Repository](https://github.com/muntazir-mehdi24/neuroplot) to submit issues or propose enhancements.

## 📄 License
NeuroPlot is open-source software licensed under the [MIT License](LICENSE).