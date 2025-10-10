<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/fal-ai-community/arbiter/blob/main/media/arbiter-dark.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/fal-ai-community/arbiter/blob/main/media/arbiter-light.png?raw=true">
  <img alt="Arbiter" src="https://github.com/fal-ai-community/arbiter/blob/main/media/arbiter-dark.png?raw=true">
</picture>
</div>

Arbiter is a Python library for evaluating and comparing generative models. It provides a unified interface for computing quality metrics across images, videos, text, and audio.

## Installation

```bash
pip install arbiter@git+https://github.com/fal-ai-community/arbiter.git
```

For development:

```bash
git clone https://github.com/fal-ai-community/arbiter.git
cd arbiter
pip install -e .
```

## Quick Start

### Python API

```python
from arbiter import Measurement, MeasurementGroup

# Single measurement
musiq = Measurement.get("musiq")
score = musiq().calculate("path/to/image.jpg")

# Multiple measurements on an image
image_metrics = MeasurementGroup.get("image")
results = image_metrics().calculate("path/to/image.jpg")

# Compare two images
lpips = Measurement.get("lpips")
distance = lpips().calculate(("reference.jpg", "generated.jpg"))

# Text-image similarity
clip_score = Measurement.get("clip_score")
similarity = clip_score().calculate(("a photo of a cat", "cat.jpg"))
```

### Command Line

```bash
# List all available measurements
arbiter list

# List measurements for specific media type
arbiter list image

# Run a single measurement
arbiter measure musiq image.jpg

# Run multiple measurements
arbiter multimeasure image.jpg
```

## Supported Measurements

### Image Measurements

#### No-Reference (Single Image Quality)

- **ARNIQA** (`arniqa`) - No-reference image quality assessment using a learned quality regressor. Returns scores in [0, 1] where higher is better.
- **CLIPIQA** (`clip_iqa`, `clipiqa`) - CLIP-based image quality assessment. Uses vision-language models to predict quality scores.
- **MUSIQ** (`musiq`) - Multi-scale Image Quality transformer. Neural network-based quality prediction, higher is better.
- **NIMA** (`nima`) - Neural Image Assessment. Predicts aesthetic and technical quality scores.
- **Variance of Laplacian** (`vol`, `variance_of_laplacian`) - Simple blur detection metric. Higher values indicate sharper images.

#### Full-Reference (Image Comparison)

- **DISTS** (`dists`, `deep_image_structure_and_texture_similarity`) - Deep Image Structure and Texture Similarity. Perceptual similarity metric using deep features.
- **LPIPS** (`lpips`, `learned_perceptual_image_patch_similarity`) - Learned Perceptual Image Patch Similarity. Lower values indicate more similar images (0 = identical).
- **MSE** (`mse`, `image_mean_squared_error`) - Mean Squared Error. Basic pixel-wise difference metric, lower is better.
- **SDI** (`sdi`, `spectral_distortion_index`) - Spectral Distortion Index. Measures distortion in frequency domain.
- **SSIM** (`ssim`, `structural_similarity_index_measure`) - Structural Similarity Index. Classic perceptual similarity metric, higher is better (max 1.0).

#### Multi-Image (Set Comparison)

- **FID** (`fid`, `frechet_inception_distance`) - Frechet Inception Distance. Measures distribution distance between two sets of images. Lower is better.
- **KID** (`kid`, `kernel_inception_distance`) - Kernel Inception Distance. Alternative to FID with better properties for small sample sizes.

#### Multi-Modal

- **CLIP Score** (`clip_score`, `clipscore`) - Measures text-image alignment using CLIP embeddings. Higher scores indicate better alignment.

### Video Measurements

Video measurements apply their image counterparts across video frames.

#### No-Reference (Single Video Quality)

- **Video ARNIQA** (`video_arniqa`) - Frame-averaged ARNIQA scores.
- **Video CLIPIQA** (`video_clip_iqa`, `video_clipiqa`) - Frame-averaged CLIP-IQA scores.
- **Video MUSIQ** (`video_musiq`) - Frame-averaged MUSIQ scores.
- **Video NIMA** (`video_nima`) - Frame-averaged NIMA scores.
- **Video Variance of Laplacian** (`video_vol`, `video_variance_of_laplacian`) - Frame-averaged blur detection.

#### Full-Reference (Video Comparison)

- **Video DISTS** (`video_dists`, `video_deep_image_structure_and_texture_similarity`) - Frame-averaged DISTS.
- **Video LPIPS** (`video_lpips`, `video_learned_perceptual_image_patch_similarity`) - Frame-averaged LPIPS.
- **Video MSE** (`video_mse`, `video_mean_squared_error`) - Frame-averaged mean squared error.
- **Video SDI** (`video_sdi`, `video_spectral_distortion_index`) - Frame-averaged spectral distortion.
- **Video SSIM** (`video_ssim`, `video_structural_similarity_index_measure`) - Frame-averaged SSIM.
- **VMAF** (`vmaf`, `video_multi_method_assessment_fusion`) - Video Multi-method Assessment Fusion. Industry-standard video quality metric.

#### Multi-Modal

- **Video CLIP Score** (`video_clip_score`, `video_clipscore`) - Frame-averaged text-video alignment scores.

### Text Measurements

- **WER** (`wer`, `word_error_rate`) - Word Error Rate. Measures transcription or translation accuracy. Lower is better.

### Audio Measurements

**In Progress** - Audio measurements are planned for future releases.

## Usage Examples

### Python API

#### Measuring Single Image Quality

```python
from arbiter import Measurement

# No-reference quality assessment
musiq = Measurement.get("musiq")
quality_score = musiq().calculate("image.jpg")
print(f"Image quality: {quality_score}")

# Check if image is blurry
vol = Measurement.get("vol")
sharpness = vol().calculate("image.jpg")
print(f"Sharpness: {sharpness}")
```

#### Comparing Two Images

```python
from arbiter import Measurement

# Perceptual similarity
lpips = Measurement.get("lpips")
distance = lpips().calculate(("reference.jpg", "generated.jpg"))
print(f"LPIPS distance: {distance}")

# Structural similarity
ssim = Measurement.get("ssim")
similarity = ssim().calculate(("reference.jpg", "generated.jpg"))
print(f"SSIM: {similarity}")
```

#### Comparing Image Distributions

```python
from arbiter import Measurement

# Frechet Inception Distance
fid = Measurement.get("fid")

# Compare two sets of images
real_images = ["real1.jpg", "real2.jpg", "real3.jpg"]
generated_images = ["gen1.jpg", "gen2.jpg", "gen3.jpg"]

# FID expects a list of tuples
image_pairs = [(r, g) for r, g in zip(real_images, generated_images)]
fid_score = fid().calculate(image_pairs)
print(f"FID: {fid_score}")
```

#### Text-Image Alignment

```python
from arbiter import Measurement

clip_score = Measurement.get("clip_score")

# Measure how well an image matches a text description
score = clip_score().calculate(("a red car on a highway", "car.jpg"))
print(f"CLIP Score: {score}")
```

#### Using Measurement Groups

```python
from arbiter import MeasurementGroup

# Get all image quality metrics
image_group = MeasurementGroup.get("image")
all_scores = image_group().calculate("image.jpg")

for metric_name, score in all_scores.items():
    print(f"{metric_name}: {score}")
```

#### Filtering Measurements by Media Type

```python
from arbiter import Measurement

# Get all image measurements
image_measurements = Measurement.for_media_type("image")

# Get all measurements that accept text and image
multimodal_measurements = Measurement.for_media_type(["text", "image"])

# Run a subset of measurements
for measurement_cls in image_measurements[:3]:
    measurement = measurement_cls()
    score = measurement.calculate(("image.jpg",))
    print(f"{measurement.name}: {score}")
```

#### Video Quality Assessment

```python
from arbiter import Measurement

# No-reference video quality
video_musiq = Measurement.get("video_musiq")
quality = video_musiq().calculate("video.mp4")
print(f"Video quality: {quality}")

# Compare two videos
video_lpips = Measurement.get("video_lpips")
distance = video_lpips().calculate(("reference.mp4", "compressed.mp4"))
print(f"Video LPIPS: {distance}")

# Industry-standard VMAF
vmaf = Measurement.get("vmaf")
vmaf_score = vmaf().calculate(("reference.mp4", "encoded.mp4"))
print(f"VMAF: {vmaf_score}")
```

#### Text Comparison

```python
from arbiter import Measurement

wer = Measurement.get("wer")
error_rate = wer().calculate(("reference transcript", "hypothesis transcript"))
print(f"Word Error Rate: {error_rate}")
```

### Command Line Interface

#### List Available Measurements

```bash
# List all measurements
arbiter list

# List only image measurements
arbiter list image

# List only video measurements
arbiter list video

# Output as JSON
arbiter list image --output-format json

# Output as CSV
arbiter list --output-format csv
```

#### Run Single Measurement

```bash
# Image quality
arbiter measure musiq image.jpg

# Image comparison
arbiter measure lpips reference.jpg generated.jpg

# Text-image alignment
arbiter measure clip_score "a photo of a dog" dog.jpg

# Video quality
arbiter measure vmaf reference.mp4 encoded.mp4

# Different output formats
arbiter measure musiq image.jpg --output-format json
arbiter measure musiq image.jpg --output-format csv
```

#### Run Multiple Measurements

```bash
# Run all compatible measurements on an image
arbiter multimeasure image.jpg

# Compare two images with all available metrics
arbiter multimeasure reference.jpg generated.jpg

# Filter measurements with regex patterns
arbiter multimeasure image.jpg --include-pattern "musiq|nima"
arbiter multimeasure image.jpg --exclude-pattern "fid|kid"

# Output options
arbiter multimeasure image.jpg --output-format json
arbiter multimeasure image.jpg --output-format markdown
```

## Architecture

Arbiter uses a plugin-based architecture where measurements are automatically discovered and registered. All measurements inherit from the `Measurement` base class and implement a standard interface.

### Key Concepts

- **Measurement**: A single metric that can be computed on media inputs
- **MeasurementGroup**: A collection of related measurements that can be executed in parallel
- **Media Types**: Measurements declare which media types they accept (image, video, text, audio)
- **Aggregate**: Some measurements operate on single inputs, others on sets of inputs

### Creating Custom Measurements

```python
from arbiter.measurements import Measurement
from arbiter.annotations import ProcessedMeasurementInputType

class CustomMetric(Measurement):
    media_type = ("image",)
    name = "custom_metric"
    aliases = ["custom"]
    
    def setup(self):
        # Initialize models/resources once
        pass
    
    def __call__(self, input: ProcessedMeasurementInputType) -> float:
        # input is a tuple of processed media (torch tensors)
        (image,) = input
        # Compute and return your metric
        return 0.0
```

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.
