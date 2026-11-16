"""Local image watermark website. Run: python web_app.py"""
import base64
import json
from io import BytesIO
from pathlib import Path
from wsgiref.simple_server import make_server

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from core import ImageWatermarkConfig, ImageWatermarker, ImageWatermarkExtractor
from core.metrics import psnr, ssim


def process(data):
    key = data.get('key', '')
    if not isinstance(key, str) or not key.strip() or len(key) > 1024:
        raise ValueError('Enter a secret key (up to 1024 characters).')
    raw = base64.b64decode(data['image'], validate=True)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('Upload an image up to 8 MB.')
    try:
        with Image.open(BytesIO(raw), formats=('PNG', 'JPEG')) as uploaded:
            if max(uploaded.size) > 2048 or min(uploaded.size) < 128:
                raise ValueError('Use an image between 128 and 2048 pixels on each side.')
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError('Upload a valid PNG or JPEG within the size limits.') from exc
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Upload a valid PNG or JPEG image.')
    h, w = image.shape[:2]
    if max(h, w) > 2048 or min(h, w) < 128:
        raise ValueError('Use an image between 128 and 2048 pixels on each side.')
    config = ImageWatermarkConfig()
    if data.get('action') == 'embed':
        text = data.get('text', '')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Enter the watermark text.')
        capacity = ImageWatermarker(config).capacity_for_shape((h, w))
        if len(text.encode()) > capacity:
            raise ValueError(f'This image holds {capacity} UTF-8 bytes. Shorten the text or use a larger image.')
        marked, _ = ImageWatermarker(config).embed_array(image, text.encode(), key, utf8=True)
        ok, encoded = cv2.imencode('.png', marked)
        if not ok:
            raise ValueError('Could not encode the output image.')
        recovered = ImageWatermarkExtractor(config).extract_array(marked, key)
        if not recovered.valid or recovered.payload != text.encode():
            raise ValueError('The watermark did not survive reconstruction on this image. Try a different or larger image.')
        return {'image': base64.b64encode(encoded).decode(), 'psnr': round(psnr(image, marked), 2),
                'ssim': round(ssim(image, marked), 4), 'capacity': capacity, 'valid': True}
    if data.get('action') == 'detect':
        result = ImageWatermarkExtractor(config).extract_array(image, key)
        return {'valid': result.valid, 'text': result.text if result.valid else None}
    raise ValueError('Unknown operation.')


def app(environ, start_response):
    route = (environ['REQUEST_METHOD'], environ['PATH_INFO'])
    if route in {('GET', '/'), ('GET', '/index.html')}:
        start_response('200 OK', [('Content-Type', 'text/html; charset=utf-8')])
        return [(Path(__file__).parent / 'web' / 'index.html').read_bytes()]
    if route == ('GET', '/health'):
        start_response('200 OK', [('Content-Type', 'text/plain')])
        return [b'ok']
    status = '200 OK'
    try:
        if route != ('POST', '/api/watermark'):
            start_response('404 Not Found', [('Content-Type', 'text/plain')])
            return [b'Not found']
        length = int(environ.get('CONTENT_LENGTH') or '0')
        if not 0 < length <= 12 * 1024 * 1024:
            raise ValueError('Upload an image up to 8 MB.')
        data = json.loads(environ['wsgi.input'].read(length))
        if not isinstance(data, dict) or not isinstance(data.get('image'), str):
            raise ValueError('Upload an image to continue.')
        result = process(data)
    except (ValueError, cv2.error) as exc:
        result, status = {'error': str(exc)}, '400 Bad Request'
    start_response(status, [('Content-Type', 'application/json'), ('Cache-Control', 'no-store')])
    return [json.dumps(result, allow_nan=False).encode()]


if __name__ == '__main__':
    print('DSP Watermark Studio: http://127.0.0.1:8000', flush=True)
    with make_server('127.0.0.1', 8000, app) as server:
        server.serve_forever()
