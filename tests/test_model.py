"""Model shape and forward-pass tests for the defect CNN."""

from __future__ import annotations

import torch

from defect_detector.models import DefectCNN


def test_defect_cnn_forward_shape():
    model = DefectCNN()
    batch = torch.rand(4, 3, 64, 64)

    output = model(batch)

    assert output.shape == (4, 2)
