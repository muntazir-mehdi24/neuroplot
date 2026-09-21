import torch
import torch.nn as nn
import matplotlib
import sys
import os

# Force a robust interactive backend for local Windows execution
if sys.platform != 'darwin' and not os.environ.get('DISPLAY'):
    pass # headless
else:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        try:
            matplotlib.use('QtAgg')
        except Exception:
            pass

import matplotlib.pyplot as plt
import numpy as np
import tempfile

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

class LiveVisualizer:
    def __init__(self, plots, model=None, data=None, update_every=50, 
                 save_gif=False, gif_name="training_progress.gif",
                 save_best_model=False, checkpoint_path="best_model.pth",
                 smooth_alpha=0.1, custom_plot_fn=None):
        self.plots = plots
        self.model = model
        self.data = data
        self.update_every = update_every
        self.save_gif = save_gif
        self.gif_name = gif_name
        self.save_best_model = save_best_model
        self.checkpoint_path = checkpoint_path
        self.smooth_alpha = smooth_alpha
        self.custom_plot_fn = custom_plot_fn
        
        self.history = {p: [] for p in plots if p not in ["boundary", "gradients", "latent", "grad_norm", "regression_fit"]}
        self.smoothed_history = {p: [] for p in plots if p not in ["boundary", "gradients", "latent", "grad_norm", "regression_fit"]}
        self.custom_plots = {}
        
        self.best_loss = float('inf')
        self.best_acc = 0.0
        
        self.frames = []
        if self.save_gif and not IMAGEIO_AVAILABLE:
            self.save_gif = False
        if self.save_gif:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.frame_count = 0
        
        self.grid_tensor = None
        self.xx = None
        self.yy = None
    
        num_plots = len(self.plots)
        cols = min(num_plots, 3)
        rows = (num_plots + cols - 1) // cols
        
        plt.ion()
        self.fig, self.axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        if num_plots == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten() if isinstance(self.axes, np.ndarray) else [self.axes]

        for j in range(num_plots, len(self.axes)):
            self.fig.delaxes(self.axes[j])

        plt.show(block=False)
        self.fig.canvas.flush_events()

        if "boundary" in self.plots and self.data is not None:
            X, _ = self.data
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100), torch.linspace(y_min, y_max, 100), indexing='ij')
            self.grid_tensor = torch.stack([xx.ravel(), yy.ravel()], dim=1)
            self.xx, self.yy = xx, yy

    def _extract_hidden_activations(self, x):
        if self.model is None:
            return None
        activation = {}
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                target_layer = module
        if target_layer is not None:
            handle = target_layer.register_forward_hook(lambda m, inp, out: activation.update({'hidden': out.detach()}))
            with torch.no_grad():
                self.model(x)
            handle.remove()
            return activation.get('hidden', None)
        return None

    def step(self, epoch, loss=None, accuracy=None, **metrics):
        if epoch % self.update_every != 0:
            return

        if loss is not None:
            if isinstance(loss, dict):
                for k, v in loss.items():
                    key_name = f"{k}_loss" if k != "loss" else "loss"
                    if key_name not in self.history:
                        self.history[key_name] = []
                        self.smoothed_history[key_name] = []
                    self.history[key_name].append(v)
                    prev = self.smoothed_history[key_name][-1] if self.smoothed_history[key_name] else v
                    self.smoothed_history[key_name].append(self.smooth_alpha * v + (1 - self.smooth_alpha) * prev)
                    if (k == "loss" or key_name == "loss") and v < self.best_loss:
                        self.best_loss = v
                        if self.save_best_model and self.model is not None:
                            torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                if "loss" not in self.history:
                    self.history["loss"] = []
                    self.smoothed_history["loss"] = []
                self.history["loss"].append(loss)
                prev = self.smoothed_history["loss"][-1] if self.smoothed_history["loss"] else loss
                self.smoothed_history["loss"].append(self.smooth_alpha * loss + (1 - self.smooth_alpha) * prev)
                if loss < self.best_loss:
                    self.best_loss = loss
                    if self.save_best_model and self.model is not None:
                        torch.save(self.model.state_dict(), self.checkpoint_path)

        if accuracy is not None:
            if "accuracy" not in self.history:
                self.history["accuracy"] = []
                self.smoothed_history["accuracy"] = []
            self.history["accuracy"].append(accuracy)
            prev = self.smoothed_history["accuracy"][-1] if self.smoothed_history["accuracy"] else accuracy
            self.smoothed_history["accuracy"].append(self.smooth_alpha * accuracy + (1 - self.smooth_alpha) * prev)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                if self.save_best_model and self.model is not None:
                    torch.save(self.model.state_dict(), self.checkpoint_path)

        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
                self.smoothed_history[key] = []
            self.history[key].append(value)
            prev = self.smoothed_history[key][-1] if self.smoothed_history[key] else value
            self.smoothed_history[key].append(self.smooth_alpha * value + (1 - self.smooth_alpha) * prev)
        
        for i, plot_type in enumerate(self.plots):
            ax = self.axes[i]
            ax.clear()
            
            if plot_type in self.history:
                ax.plot(self.history[plot_type], color='gray', alpha=0.3, lw=1, label='Raw')
                ax.plot(self.smoothed_history[plot_type], color='red' if 'loss' in plot_type else 'green', lw=2, label='EMA')
                ax.set_title(f"{plot_type.capitalize()} Curve")
                ax.grid(True, linestyle='--', alpha=0.6)

            elif plot_type == "boundary" and self.model is not None and self.data is not None:
                X, y = self.data
                if self.grid_tensor is not None:
                    self.model.eval()
                    with torch.no_grad():
                        Z = self.model(self.grid_tensor).detach().cpu().numpy()
                    self.model.train()
                    ax.contourf(self.xx, self.yy, Z.reshape(self.xx.shape), levels=50, cmap=plt.cm.Spectral, alpha=0.7)
                ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.Spectral, edgecolors='k')
                ax.set_title(f"Decision Boundary (Epoch {epoch})")

            elif plot_type == "latent" and self.model is not None and self.data is not None:
                X, y = self.data
                h = self._extract_hidden_activations(X)
                if h is not None:
                    h_np = h.cpu().numpy()
                    ax.scatter(h_np[:, 0], h_np[:, 1] if h_np.shape[1] > 1 else np.zeros_like(h_np[:, 0]), c=y.squeeze(), cmap=plt.cm.Spectral, edgecolors='k')
                    ax.set_title(f"Latent Space (Epoch {epoch})")

            elif plot_type == "grad_norm" and self.model is not None:
                total_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
                if not hasattr(self, 'grad_norms'):
                    self.grad_norms = []
                self.grad_norms.append(total_norm)
                ax.plot(self.grad_norms, color="purple", lw=2)
                ax.set_title("Total Gradient Norm")
                ax.grid(True, linestyle='--', alpha=0.6)

            elif plot_type == "regression_fit" and self.model is not None and self.data is not None:
                X, y = self.data
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(X).detach().cpu().numpy().squeeze()
                self.model.train()
                y_np = y.cpu().numpy().squeeze()
                ax.scatter(y_np, preds, color="teal", alpha=0.7, edgecolors='k')
                m_val = min(y_np.min(), preds.min())
                mx_val = max(y_np.max(), preds.max())
                ax.plot([m_val, mx_val], [m_val, mx_val], 'r--', label="y=x")
                ax.set_title("Regression: Actual vs Predicted")
                ax.legend(fontsize=7)
                ax.grid(True, linestyle='--', alpha=0.6)

            elif plot_type in self.custom_plots:
                self.custom_plots[plot_type](ax, self.model, self.data)
            elif self.custom_plot_fn is not None and plot_type == "custom":
                self.custom_plot_fn(ax, self.model, self.data)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

        if self.save_gif and IMAGEIO_AVAILABLE:
            frame_path = os.path.join(self.temp_dir.name, f"frame_{self.frame_count:04d}.png")
            self.fig.savefig(frame_path, dpi=100)
            self.frames.append(frame_path)
            self.frame_count += 1

    def close(self):
        plt.ioff()
        if self.save_best_model:
            print(f"Best model saved to: {os.path.abspath(self.checkpoint_path)}")
        if self.save_gif and IMAGEIO_AVAILABLE and self.frames:
            print("Compiling training progress into GIF...")
            images = [imageio.v3.imread(f) for f in self.frames]
            imageio.mimsave(self.gif_name, images, duration=200, loop=0)
            self.temp_dir.cleanup()