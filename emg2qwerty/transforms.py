# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )


@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data

@dataclass
class RandomScaling:   
    min_factor: float = 0.98
    max_factor: float = 1.02
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = np.random.uniform(self.min_factor, self.max_factor)
        return tensor * scale
    
@dataclass
class RandomGaussianNoise:
    mean: float = 0.0
    std: float = 0.005

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:        
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

@dataclass    
class RandomChannelDropout:
    p: float = 0.1
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        num_channels = tensor.shape[self.channel_dim]
        dropout_mask = torch.rand(num_channels) > self.p
        mask_shape = [1] * tensor.ndim
        mask_shape[self.channel_dim] = num_channels
        #dropout_mask = dropout_mask.view(mask_shape).to(tensor.device)

        return tensor * dropout_mask


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)


@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16
    n_mel: int = 6

    '''def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            n_mel = self.n_mel,
            center=False,
        )'''
    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(

            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.

            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)

@dataclass
class VariableHopLength:
    """Creates a log10-scaled spectrogram with variable hop length.

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
        hop_length_range (tuple): Range of hop lengths.
    """
    n_fft: int = 64
    hop_length_range: tuple[int, int] = (8, 32)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        hop_length = np.random.randint(self.hop_length_range[0], self.hop_length_range[1] + 1)
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=hop_length,
            normalized=True,
            center=False,
        )
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)

@dataclass
class VariableFFT:
    """Creates a log10-scaled spectrogram with variable FFT size.

    Args:
        n_fft_range (tuple): Range of FFT.
        hop_length (int): Number of samples to stride between consecutive STFT windows.
    """
    n_fft_range: tuple[int, int] = (32, 128)
    hop_length: int = 16

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        n_fft = np.random.randint(self.n_fft_range[0], self.n_fft_range[1] + 1)
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
        )
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)

@dataclass
class SpatialWarping:
    """Applies spatial warping to the EMG signals.

    Args:
        max_warp (float): Maximum warp factor for spatial warping.
    """
    max_warp: float = 0.1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        warp = torch.FloatTensor(tensor.shape[-1]).uniform_(-self.max_warp, self.max_warp)
        tensor_new = tensor.clone()
        for i in range(tensor.shape[-1]):
            tensor_new[..., i] = tensor[..., i] * (1 + warp[i])
        return tensor_new

@dataclass
class FrequencyWarping:
    """Applies frequency warping augmentation.

    Args:
        max_warp (float): Maximum warp factor.
    """
    max_warp: float = 0.1

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        warp_factor = np.random.uniform(-self.max_warp, self.max_warp)
        freq_bins = spec.size(-2)
        warp_indices = torch.arange(freq_bins, dtype=torch.float32) * (1 + warp_factor)
        warp_indices = torch.clamp(warp_indices, 0, freq_bins - 1).long()
        return spec.index_select(-2, warp_indices)

@dataclass
class RunningTimeNormalization:
    """Applies running time normalization to the EMG signals.

    Args:
        epsilon (float): Small value to avoid division by zero.
    """
    epsilon: float = 1e-6

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mu = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True) + self.epsilon
        tensor_new = (tensor - mu) / std
        return tensor_new
