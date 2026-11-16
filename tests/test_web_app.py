import base64
from pathlib import Path

from web_app import process


def test_web_roundtrip_and_wrong_key():
    source = Path('report/figures/nature_original.png').read_bytes()
    marked = process({'action': 'embed', 'image': base64.b64encode(source).decode(),
                      'key': 'web-test-key', 'text': 'DSP demo'})
    assert marked['valid']
    recovered = process({'action': 'detect', 'image': marked['image'], 'key': 'web-test-key'})
    assert recovered == {'valid': True, 'text': 'DSP demo'}
    wrong = process({'action': 'detect', 'image': marked['image'], 'key': 'wrong-key'})
    assert wrong == {'valid': False, 'text': None}
