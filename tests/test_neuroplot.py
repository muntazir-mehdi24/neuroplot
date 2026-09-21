import matplotlib
matplotlib.use('Agg')

import pytest
import torch
import torch.nn as nn
import numpy as np
from neuroplot import LiveVisualizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    def forward(self, x):
        return self.linear(x)

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
    def forward(self, x):
        return self.conv(x)

def test_validation_errors():
    # 1. 'boundary' with missing model
    with pytest.raises(ValueError, match="requires a model"):
        LiveVisualizer(plots=["boundary"], data=(torch.randn(10, 2), torch.randn(10, 1)))

    # 2. 'boundary' with wrong feature shape (e.g. 5 features instead of 2)
    model = SimpleModel()
    bad_data = (torch.randn(10, 5), torch.randn(10, 1))
    with pytest.raises(ValueError, match="requires 2D input data with 2 features"):
        LiveVisualizer(plots=["boundary"], model=model, data=bad_data)

    # 3. 'latent' with a model lacking any nn.Linear layer
    conv_model = ConvModel()
    with pytest.raises(ValueError, match="requires the model to contain at least one nn.Linear layer"):
        LiveVisualizer(plots=["latent"], model=conv_model, data=(torch.randn(10, 1, 5, 5), torch.randn(10, 1)))


def test_scalar_and_dict_loss_handling():
    model = SimpleModel()
    viz = LiveVisualizer(plots=["loss"], model=model, update_every=1)
    
    # Test scalar loss step
    viz.step(epoch=1, loss=2.5)
    assert "loss" in viz.history
    assert viz.history["loss"][-1] == 2.5

    # Test dictionary loss step
    viz.step(epoch=2, loss={"ce": 1.2, "kl": 0.4})
    assert "ce_loss" in viz.history
    assert "kl_loss" in viz.history
    assert viz.history["ce_loss"][-1] == 1.2
    assert viz.history["kl_loss"][-1] == 0.4
    
    viz.close()


def test_ema_smoothing_math():
    model = SimpleModel()
    viz = LiveVisualizer(plots=["loss"], model=model, smooth_alpha=0.5, update_every=1)
    
    # Step 1: initial loss = 10.0 -> smoothed should be 10.0
    viz.step(epoch=1, loss=10.0)
    assert viz.smoothed_history["loss"][0] == 10.0
    
    # Step 2: loss = 2.0 -> smoothed = 0.5 * 2.0 + (0.5 * 10.0) = 6.0
    viz.step(epoch=2, loss=2.0)
    assert viz.smoothed_history["loss"][1] == 6.0
    
    viz.close()


def test_grid_layout_handling():
    model = SimpleModel()
    # Test 1 plot (should handle single axis conversion)
    viz1 = LiveVisualizer(plots=["loss"], model=model)
    assert len(viz1.axes) == 1
    viz1.close()

    # Test 4 plots (should layout in grid format)
    viz4 = LiveVisualizer(plots=["loss", "grad_norm", "regression_fit", "loss"], model=model, data=(torch.randn(10, 2), torch.randn(10, 1)))
    assert len(viz4.axes) == 4
    viz4.close()