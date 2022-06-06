import ao
import torch
import numpy as np

from typing import List, Optional


class AO:

    def __init__(
        self,
        model_path: Path,
        extractors: List[ao.extractor.Extractor],
        num_frames: int,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        ):
        self.model_path = model_path
        if not len(extractors):
            raise ValueError('At least one extractor must be provided')
        self.sample_rate = extractors[0].sample_rate
        self.num_samples = extractors[0].num_samples
        self.num_features = extractors[0].num_features
        for extractor in extractors:
            if extractor.num_samples != self.num_samples:
                raise ValueError(
                    "Provided extractors with different number of input "
                    f"samples: {extractor.num_samples} != {self.num_samples}"
                    )
            if extractor.num_features != self.num_features:
                raise ValueError(
                    "Provided extractors with different number of output "
                    f"features: {extractor.num_features} != {self.num_features}"
                    )
        self.num_frames = num_frames
        self.device = device
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.features = torch.empty(
            [1, len(extractors), self.num_features, self.num_frames],
            device=self.device,
            )
        # TODO allocate with number of labels
        self.prediction = torch.empty(6, device=self.device)

    def update(self, samples: np.ndarray) -> torch.Tensor:
        for i, extractor in enumerate(extractors):
            self.features[0, i, :, 0] = torch.as_tensor(extractor(samples))
        self.features.roll(-1, 3)
        return self.features

    def predict(self, samples: Optional[np.ndarray] = None) -> torch.Tensor:
        if samples is not None:
            self.update(samples)
        self.prediction = self.model(self.features)
        return self.prediction

    def __call__(self, samples: np.ndarray):
        self.predict(samples)
        return self.prediction.argmax(1).sum().item()