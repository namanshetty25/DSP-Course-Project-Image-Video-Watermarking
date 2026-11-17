"""Small, display-only snapshots of the actual image transforms."""
import base64

import cv2
import numpy as np

from .media import split_luminance
from .transforms import dwt2, pad_to_multiple, blocks_view, dct2_blocks, merge_blocks


def panel(name, values, signed=False):
    values = np.asarray(values, dtype=float)
    if signed:
        scale = max(float(np.max(np.abs(values))), 1e-9)
        display = 127.5 + values * (127.5 / scale)
    else:
        display = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX)
    display = np.clip(display, 0, 255).astype(np.uint8)
    display = cv2.resize(display, (192, 192), interpolation=cv2.INTER_NEAREST)
    _, encoded = cv2.imencode('.png', display)
    return {'name': name, 'image': base64.b64encode(encoded).decode(),
            'shape': list(values.shape), 'minimum': round(float(values.min()), 3),
            'maximum': round(float(values.max()), 3)}


def transform_stages(image, config):
    luminance, _ = split_luminance(image)
    stages = [{'title': 'Brightness channel',
               'description': 'Extract Y from YCrCb. The watermark changes brightness while retaining color information.',
               'panels': [panel('Y luminance', luminance)]}]
    approximation, _ = pad_to_multiple(luminance, config.spatial_multiple)
    for level in range(1, config.dwt_level + 1):
        approximation, details = dwt2(approximation, config.wavelet, config.wavelet_mode)
        stages.append({'title': f'Haar decomposition · level {level}',
                       'description': 'Low-pass and high-pass filters split the image into four bands, then downsample each axis. The LL band feeds the next level. Detail images use gray for zero, light for positive and dark for negative coefficients.',
                       'panels': [panel('LL approximation', approximation)] +
                       [panel(name, band, True) for name, band in zip(('LH detail', 'HL detail', 'HH detail'), details)]})
    coefficients = dct2_blocks(blocks_view(details[0], config.block_size))
    stages.append({'title': 'Block DCT · selected detail band',
                   'description': 'Each 4 × 4 LH block becomes frequency coefficients. The embedder uses non-DC coefficients from both LH and HL at level 3. This heatmap shows the LH coefficients, normalized for display.',
                   'panels': [panel('LH₃ block DCT', merge_blocks(coefficients), True)]})
    return stages
