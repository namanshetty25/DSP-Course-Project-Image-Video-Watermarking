"""Local image watermark website. Run: python web_app.py"""
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from core import ImageWatermarkConfig, ImageWatermarker, ImageWatermarkExtractor
from core.metrics import psnr, ssim


def process(data):
    key = data.get('key', '')
    if not isinstance(key, str) or not key.strip() or len(key) > 1024:
        raise ValueError('Enter a secret key (up to 1024 characters).')
    raw = base64.b64decode(data['image'], validate=True)
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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/', '/index.html'):
            self.send_error(404)
            return
        content = (Path(__file__).parent / 'web' / 'index.html').read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path != '/api/watermark':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 12 * 1024 * 1024:
                raise ValueError('Upload size exceeds the 8 MB image limit.')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Invalid request.')
            result = process(data)
            status = 200
        except (ValueError, KeyError, TypeError, cv2.error) as exc:
            result, status = {'error': str(exc)}, 400
        except Exception:
            result, status = {'error': 'Processing failed. Try a different image.'}, 500
        content = json.dumps(result).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)


if __name__ == '__main__':
    print('DSP Watermark Studio: http://127.0.0.1:8000', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
