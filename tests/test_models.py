import torch

import spineloc.models.spine


def test_generate_coordinate_targets():
    coords = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.5, 0.5, 0.75, 0.75],
        ]
    )  # Shape: (B=2, 4)
    feature_map_size = (4, 4)

    expected_targets = torch.tensor(
        [
            [
                [
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3333, 0.3333, 0.3333, 0.3333],
                    [0.6667, 0.6667, 0.6667, 0.6667],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ],
                [
                    [0.0000, 0.3333, 0.6667, 1.0000],
                    [0.0000, 0.3333, 0.6667, 1.0000],
                    [0.0000, 0.3333, 0.6667, 1.0000],
                    [0.0000, 0.3333, 0.6667, 1.0000],
                ],
            ],
            [
                [
                    [0.5000, 0.5000, 0.5000, 0.5000],
                    [0.5833, 0.5833, 0.5833, 0.5833],
                    [0.6667, 0.6667, 0.6667, 0.6667],
                    [0.7500, 0.7500, 0.7500, 0.7500],
                ],
                [
                    [0.5000, 0.5833, 0.6667, 0.7500],
                    [0.5000, 0.5833, 0.6667, 0.7500],
                    [0.5000, 0.5833, 0.6667, 0.7500],
                    [0.5000, 0.5833, 0.6667, 0.7500],
                ],
            ],
        ]
    )  # Shape: (B=2, 2, 4, 4)

    targets = spineloc.models.spine.generate_coordinate_targets(coords, feature_map_size)

    assert torch.allclose(targets, expected_targets, atol=1e-4), (
        "Generated targets do not match expected values."
    )
