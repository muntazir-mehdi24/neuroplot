# import necessary libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
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
                 smooth_alpha=0.1):
        self.plots = plots
        self.model = model
        self.data = data
        self.update_every = update_every
        self.save_gif = save_gif
        self.gif_name = gif_name
        self.save_best_model = save_best_model
        self.checkpoint_path = checkpoint_path
        self.smooth_alpha = smooth_alpha
        
        # Dynamic history dictionary
        self.history = {plot_type: [] for plot_type in plots if plot_type not in ["boundary", "gradients", "latent"]}
        self.smoothed_history = {plot_type: [] for plot_type in plots if plot_type not in ["boundary", "gradients", "latent"]}
        self.custom_plots = {}
        
        # Best metric trackers
        self.best_loss = float('inf')
        self.best_acc = 0.0
        
        # GIF frame storage
        self.frames = []
        if self.save_gif and not IMAGEIO_AVAILABLE:
            print("Warning: 'imageio' package not found. GIF saving will be skipped. Install via 'pip install imageio'.")
            self.save_gif = False

        if self.save_gif:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.frame_count = 0
        
        # Grid precomputation for decision boundary
        self.grid_tensor = None
        self.xx = None
        self.yy = None
    
        # Set up dynamic subplot grid
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

        if "boundary" in self.plots and self.data is not None:
            X, _ = self.data
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100), torch.linspace(y_min, y_max, 100), indexing='ij')
            self.grid_tensor = torch.stack([xx.ravel(), yy.ravel()], dim=1)
            self.xx, self.yy = xx, yy

    def add_custom_plot(self, name, func):
        self.custom_plots[name] = func

    def _extract_hidden_activations(self, x):
        """Helper to extract hidden layer representations for mechanistic tasks."""
        if self.model is None:
            return None
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # Automatically find the second-to-last linear layer or hook a fallback
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                target_layer = module
        
        if target_layer is not None:
            handle = target_layer.register_forward_hook(get_activation('hidden'))
            with torch.no_grad():
                self.model(x)
            handle.remove()
            return activation.get('hidden', None)
        return None

    def step(self, epoch, **metrics):
        if epoch % self.update_every != 0:
            return

        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
                prev_smooth = self.smoothed_history[key][-1] if self.smoothed_history[key] else value
                smoothed_val = self.smooth_alpha * value + (1 - self.smooth_alpha) * prev_smooth
                self.smoothed_history[key].append(smoothed_val)
                
                if key == "loss" and value < self.best_loss:
                    self.best_loss = value
                    if self.save_best_model and self.model is not None:
                        torch.save(self.model.state_dict(), self.checkpoint_path)
                elif key == "accuracy" and value > self.best_acc:
                    self.best_acc = value
                    if self.save_best_model and self.model is not None:
                        torch.save(self.model.state_dict(), self.checkpoint_path)
        
        for i, plot_type in enumerate(self.plots):
            ax = self.axes[i]
            ax.clear()
            
            # 1. Metric Curves with EMA Trendlines
            if plot_type in self.history:
                raw_values = self.history[plot_type]
                smooth_values = self.smoothed_history[plot_type]
                
                ax.plot(raw_values, color='gray', alpha=0.3, lw=1, label='Raw')
                ax.plot(smooth_values, color='red' if plot_type == 'loss' else 'green', lw=2, label='EMA Trend')
                ax.set_title(f"{plot_type.capitalize()} Curve")
                ax.set_xlabel("Update Step")
                ax.set_ylabel(plot_type.capitalize())
                ax.grid(True, linestyle='--', alpha=0.6)
                
                if plot_type == "loss" and raw_values:
                    best_idx = np.argmin(raw_values)
                    ax.scatter(best_idx, raw_values[best_idx], color='gold', s=100, zorder=5, label=f'Best: {raw_values[best_idx]:.4f}')
                    ax.legend(loc='upper right', fontsize=7)
                elif plot_type == "accuracy" and raw_values:
                    best_idx = np.argmax(raw_values)
                    ax.scatter(best_idx, raw_values[best_idx], color='gold', s=100, zorder=5, label=f'Best: {raw_values[best_idx]:.4f}')
                    ax.legend(loc='lower right', fontsize=7)

            # 2. Decision Boundary Plot
            elif plot_type == "boundary" and self.model is not None and self.data is not None:
                X, y = self.data
                if self.grid_tensor is not None:
                    self.model.eval()
                    with torch.no_grad():
                        Z = self.model(self.grid_tensor).detach().cpu().numpy()
                    self.model.train()
                    Z = Z.reshape(self.xx.shape)
                    ax.contourf(self.xx, self.yy, Z, levels=50, cmap=plt.cm.Spectral, alpha=0.7)
                
                ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.Spectral, edgecolors='k')
                ax.set_title(f"Decision Boundary (Epoch {epoch})")

            # 3. Mechanistic Latent Space / Internal Representation Warp Plot
            elif plot_type == "latent" and self.model is not None and self.data is not None:
                X, y = self.data
                hidden_acts = self._extract_hidden_activations(X)
                if hidden_acts is not None:
                    h = hidden_acts.cpu().numpy()
                    # If hidden dimension > 2, take the first 2 principal components or dimensions for visualization
                    h_x = h[:, 0]
                    h_y = h[:, 1] if h.shape[1] > 1 else np.zeros_like(h_x)
                    
                    ax.scatter(h_x, h_y, c=y.squeeze(), cmap=plt.cm.Spectral, edgecolors='k', s=40)
                    ax.set_title(f"Internal Latent Space (Epoch {epoch})")
                    ax.set_xlabel("Hidden Dim 1")
                    ax.set_ylabel("Hidden Dim 2")
                    ax.grid(True, linestyle='--', alpha=0.6)

            # 4. Gradient Flow Plot
            elif plot_type == "gradients" and self.model is not None:
                ave_grads = []
                layers = []
                for n, p in self.model.named_parameters():
                    if p.requires_grad and ("bias" not in n) and (p.grad is not None):
                        layers.append(n.split(".")[0])
                        ave_grads.append(p.grad.abs().mean().item())
                
                if ave_grads:
                    ax.plot(ave_grads, alpha=0.7, color="blue", lw=2)
                    ax.bar(range(len(ave_grads)), ave_grads, alpha=0.3, color="blue")
                    ax.set_xticks(range(len(layers)))
                    ax.set_xticklabels(layers, rotation=30, ha='right', fontsize=8)
                    ax.set_title("Gradient Flow (Layer-wise)")
                    ax.set_xlabel("Layers")
                    ax.set_ylabel("Avg Gradient Magnitude")
                    ax.set_yscale("log")
                    ax.grid(True, linestyle='--', alpha=0.6)

            # 5. Custom Callbacks
            elif plot_type in self.custom_plots:
                self.custom_plots[plot_type](ax, self.model, self.data)

        plt.tight_layout()
        plt.draw()
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
            abs_ckpt = os.path.abspath(self.checkpoint_path)
            print(f"Best model automatically saved to:\n---> {abs_ckpt}")

        if self.save_gif and IMAGEIO_AVAILABLE and self.frames:
            abs_path = os.path.abspath(self.gif_name)
            print(f"Compiling training progress into GIF...")
            images = [imageio.v3.imread(f) for f in self.frames]
            imageio.mimsave(self.gif_name, images, duration=200, loop=0)
            self.temp_dir.cleanup()
            print(f"GIF successfully compiled and saved at:\n---> {abs_path}")

        plt.show()