# Image and Video Watermarking

A Digital Signal Processing course project for embedding and recovering invisible, keyed watermarks in images and videos. Built with Python, NumPy, SciPy, PyWavelets, OpenCV, and FFmpeg.

![Original and watermarked image](assets/original_vs_watermarked.png)

## How it works

The image pipeline modifies luminance using **multi-level DWT → block DCT → keyed STDM** (spread-transform dither modulation), then reconstructs the image. An optional SVD-QIM backend is available for comparison. Payloads use convolutional error correction, CRC checks, and HMAC authentication.

Video watermarking adds a temporal Haar transform and combines extraction scores across multiple frames. Image extraction is blind by default; geometric alignment can use the original image as a reference.

For the course-project explanation, see [DSP theory and implementation](docs/DSP_THEORY.md): the original mathematical derivations, embedding/extraction logic, video processing, and links to the Python modules.

The [course report](report/README.md) includes editable LaTeX, result figures, measured tables, and a script to reproduce the experiments.

## Setup

### Interactive image website

After installing the project dependencies, run `python web_app.py` from the repository root and open <http://127.0.0.1:8000>.
Use **Add watermark** to upload a PNG/JPEG, enter hidden text and a secret key, and download a verified watermarked PNG.
Use **Detect watermark** to upload that PNG and recover the authenticated text with the same key.
The app uses the default DSP configuration and processes images in memory. The local command binds to your computer.

Requires **Python 3.10+**. Video operations also require `ffmpeg` and `ffprobe` on your `PATH`, with the encoder you select (`libx264` by default).

```bash
git clone https://github.com/namanshetty25/DSP-Course-Project-Image-Video-Watermarking.git
cd DSP-Course-Project-Image-Video-Watermarking
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install the project and its dependencies:

```bash
python -m pip install -e .
python cli.py --help
```

All dependencies are defined in `pyproject.toml`. The installed `dsp-watermark` command is equivalent to `python cli.py`.

## Image quickstart

Set a key (at least 8 bytes). Use the same key when embedding and extracting; the value below is only a demonstration.

```powershell
# Windows PowerShell
$env:DSP_WATERMARK_KEY = 'replace-this-demo-key'
```

```bash
# Linux / macOS
export DSP_WATERMARK_KEY='replace-this-demo-key'
```

Embed a short message in the included 512×512 sample, then recover it:

```bash
python cli.py image embed --input assets/original.png --output output/marked.png --message "asset-id:42"
python cli.py image extract --input output/marked.png --manifest output/marked.png.wm.json
```

Embedding creates a sidecar `.wm.json` manifest containing configuration and extraction metadata. Keep it with the marked file. It contains neither the key nor the plaintext payload. Successful extraction reports `valid: true` and the recovered text.

Useful options:

- Use `--payload-file metadata.bin` instead of `--message` for a binary payload.
- Use `--key-file path/to/key` to read a key from a file.
- Add `--output-payload output/recovered.bin` when extracting binary data.
- Add `--reference assets/original.png` when extracting from a rotated or scaled image.
- Prefer PNG output to avoid introducing compression during the first save.

Check capacity before embedding larger messages:

```bash
python cli.py image capacity --width 512 --height 512
```

The default 512×512 configuration holds 40 payload bytes. Capacity depends on image dimensions, transform settings, and coding overhead.

For a Python API round trip:

```bash
python -m examples.quickstart
```

## Video quickstart

With FFmpeg installed and the key set, generate a sample video, embed, and extract:

```bash
python -m examples.generate_sample_video --output output/sample-input.avi
python cli.py video embed --input output/sample-input.avi --output output/marked.mp4 --message "asset-id:42"
python cli.py video extract --input output/marked.mp4 --manifest output/marked.mp4.wm.json
```

Replace the sample input with your own video as needed. Use `--codec libx265` for H.265 output, or `--crf` to adjust encoding quality. Run `python cli.py video embed --help` for all options.

## Benchmarks

The attack suites test compression, noise, filtering, geometric changes, and temporal video changes. Generate image figures and JSON/CSV/Markdown reports locally:

```bash
python generate_assets.py
```

Add `--include-video` to run video benchmarks, or `--output-dir results/figures` to choose a separate output directory. Generated outputs are ignored by Git; only the sample cover and README preview are tracked.

To benchmark your own image using the configured key:

```bash
python cli.py image benchmark --input assets/original.png --message "asset-id:42" --json-output results/image-benchmark.json
```

See [benchmark methodology](docs/BENCHMARKING.md) for fixtures, attacks, and metric definitions, and [algorithm details](docs/ALGORITHM.md) for transform conventions, capacity calculations, and decoding.

## Project structure

```text
core/                 Watermark embedding, extraction, transforms, and media I/O
attacks/              Spatial, compression, and temporal attacks
benchmarks/           Reproducible fixtures and benchmark runners
examples/             Image API example and sample-video generator
tests/                Unit and video integration tests
docs/                 Algorithm and benchmark references
assets/               Sample cover and README preview
.github/workflows/    Continuous integration
cli.py                Command-line interface
generate_assets.py    Figure and benchmark report generator
pyproject.toml        Dependencies, packaging, and tool configuration
MANIFEST.in           Supporting files included in source distributions
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -m "not integration"
python -m pytest -m integration
python -m build
```

Video integration tests require FFmpeg. CI runs lint and unit tests across Python 3.10, 3.12, and 3.13, with video tests, asset generation, and packaging on Python 3.12.

## Limitations

Recovery depends on media content, payload size, strength settings, and attack severity. Geometric changes may require the original reference; severe cropping or repeated lossy encoding can prevent recovery. Results on the synthetic fixture do not establish performance across all images or videos. HMAC authenticates the embedded payload; it does not encrypt the media or establish ownership by itself.

## License

[MIT](LICENSE).
