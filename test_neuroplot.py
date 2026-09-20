import torch
import torch.nn as nn
import torch.optim as optim
from neuroplot import LiveVisualizer

# Simple regression model to test multiple features at once
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(200, 2)
y = X[:, 0:1] + 2 * X[:, 1:2] + 0.1 * torch.randn(200, 1)

model = Regressor()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Custom plot hook function
def custom_residual_plot(ax, model, data):
    X_val, y_val = data
    model.eval()
    with torch.no_grad():
        preds = model(X_val)
        residuals = (y_val - preds).cpu().numpy()
    model.train()
    ax.hist(residuals, bins=15, color='orange', edgecolor='black', alpha=0.7)
    ax.set_title("Custom Hook: Residuals")
    ax.grid(True, linestyle='--', alpha=0.6)

# Initialize LiveVisualizer with new features
viz = LiveVisualizer(
    plots=["loss", "grad_norm", "regression_fit", "custom"],
    model=model,
    data=(X, y),
    update_every=5,
    save_gif=True,
    gif_name="regression_test.gif",
    save_best_model=True,
    checkpoint_path="best_regressor.pth",
    custom_plot_fn=custom_residual_plot
)

print("Running test training loop...")
for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    preds = model(X)
    
    train_loss = criterion(preds, y)
    val_loss = train_loss * 1.15
    
    train_loss.backward()
    optimizer.step()
    
    # Test dictionary loss input
    viz.step(
        epoch=epoch, 
        loss={"train": train_loss.item(), "val": val_loss.item()}
    )

viz.close()
print("Test finished successfully!")