# KYC Verification Pipeline

A Python-based Know Your Customer (KYC) verification system that uses facial recognition to match ID documents with live webcam captures.

## Features

- **Live Webcam Capture**: Real-time face capture with countdown timer
- **Face Detection**: Uses MTCNN for accurate face detection
- **Face Matching**: DeepFace integration with multiple model support (VGG-Face, ArcFace, etc.)
- **Image Enhancement**: Advanced preprocessing including upscaling, denoising, and contrast enhancement
- **Risk Assessment**: Automated risk scoring and approval decisions
- **Comprehensive Reporting**: Detailed verification reports with confidence scores

## Requirements

```bash
opencv-python-headless
pillow
numpy
pytesseract
passporteye
deepface
tensorflow
mtcnn
imutils
```

### System Dependencies

For Ubuntu/Debian:
```bash
apt-get install tesseract-ocr libtesseract-dev
```

## Installation

```bash
pip install opencv-python-headless pillow numpy pytesseract passporteye
pip install deepface tensorflow mtcnn imutils
```

## Usage

### Basic Example

```python
from KYCpipeline import BankingKYCPipeline

# Initialize the pipeline
kyc = BankingKYCPipeline()

# Run verification
result = kyc.verify_customer(
    id_document_path='path/to/id_document.jpg',
    model='ArcFace',
    enhance_document=True,
    enhance_webcam=True,
    countdown=3
)

# Generate report
print(kyc.generate_report(result))

# Save results
kyc.save_result(result, 'kyc_result.json')
```

## Components

### FaceProcessor
Handles face detection, extraction, and enhancement from images.

### WebcamFaceMatcher
Manages webcam capture and face matching with configurable confidence thresholds.

### BankingKYCPipeline
Complete verification workflow including risk assessment and decision-making.

## Configuration

- **Minimum Confidence Threshold**: 10% (configurable in `WebcamFaceMatcher.MIN_CONFIDENCE_THRESHOLD`)
- **Supported Models**: VGG-Face, ArcFace, Facenet, and others supported by DeepFace
- **Enhancement Options**: Optional preprocessing for both document and webcam images

## Output

The verification results include:
- Match status (MATCH/NO MATCH)
- Confidence percentage
- Distance score
- Risk assessment score
- Approval decision
- Detection quality metrics

## Notes

- Designed for Google Colab environment
- Requires webcam access for live capture
- Results are saved in JSON format for audit purposes
- Matches below the minimum confidence threshold are automatically rejected

## License

This project is intended for educational and development purposes.
