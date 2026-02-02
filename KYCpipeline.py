# ============================================================================
# Cell 1: Install dependencies
# ============================================================================

!pip install opencv-python-headless pillow numpy pytesseract passporteye
!pip install deepface tensorflow mtcnn imutils

# For Colab, also run:
!apt-get install install tesseract-ocr libtesseract-dev

# ============================================================================
# Cell 2: Import dependencies
# ============================================================================

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from deepface import DeepFace
from mtcnn import MTCNN
from typing import Optional, Tuple, Dict
from datetime import datetime
import json
import warnings

# For webcam in Colab
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

warnings.filterwarnings('ignore')

# ============================================================================
# Cell 3: FACE PROCESSOR
# ============================================================================

class FaceProcessor:
    """Detect, extract and enhance faces from images."""

    def __init__(self):
        self.detector = MTCNN()
        print("✅ Face Processor initialized")

    def detect_faces(self, image_path: str) -> Tuple:
        """Detect all faces in an image."""
        img = cv2.imread(image_path)
        if img is None:
            return None, None

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_img)
        return faces, img

    def extract_face(self, image_path: str, padding: int = 20) -> Optional[Tuple]:
        """Extract the best quality face from image."""
        faces, img = self.detect_faces(image_path)

        if not faces or len(faces) == 0:
            return None

        best_face = max(faces, key=lambda x: x['confidence'])

        # Extract with padding
        x, y, w, h = best_face['box']
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding

        face_img = img[y:y+h, x:x+w]
        return face_img, best_face

    def enhance_face(self, face_img: np.ndarray,
                    target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Enhance face image quality for better matching."""
        h, w = face_img.shape[:2]

        upscaled = cv2.resize(face_img, (w*4, h*4), interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)

        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Resize to target
        final = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_LANCZOS4)
        return final

# ============================================================================
# Cell 4: WEBCAM FACE MATCHER
# ============================================================================

class WebcamFaceMatcher:
    # CONFIDENCE THRESHOLD FOR MATCHING
    MIN_CONFIDENCE_THRESHOLD = 10.0

    def __init__(self, face_processor):
        self.face_processor = face_processor
        print("✅ Webcam Face Matcher initialized")

    def capture_from_webcam(self, save_path: str = '/content/webcam_capture.jpg',
                           countdown: int = 3) -> Optional[str]:
        """Capture image from webcam with countdown timer."""
        print(f"📸 Starting webcam capture (countdown: {countdown}s)...")

        js_code = Javascript('''
        async function takePhoto(quality, countdown) {
            const div = document.createElement('div');
            div.style.cssText = 'position: relative; width: 640px; margin: 20px auto;';
            div.id = 'camera-container';

            const video = document.createElement('video');
            video.style.cssText = 'width: 100%; border: 3px solid #4CAF50; border-radius: 10px;';
            video.autoplay = true;
            video.playsinline = true;

            const canvas = document.createElement('canvas');
            div.appendChild(video);
            document.body.appendChild(div);

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
                });

                video.srcObject = stream;

                await new Promise((resolve, reject) => {
                    video.onloadedmetadata = () => {
                        video.play().then(() => {
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            resolve();
                        }).catch(reject);
                    };
                    setTimeout(() => reject(new Error("Timeout")), 10000);
                });

                await new Promise(resolve => setTimeout(resolve, 1000));

                // Countdown overlay
                const countdownDiv = document.createElement('div');
                countdownDiv.style.cssText = `
                    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                    font-size: 120px; color: #FF5722; font-weight: bold;
                    text-shadow: 3px 3px 6px rgba(0,0,0,0.7); z-index: 9999;
                    background: rgba(255,255,255,0.3); padding: 20px; border-radius: 20px;
                `;
                div.appendChild(countdownDiv);

                for(let i = countdown; i > 0; i--) {
                    countdownDiv.textContent = i;
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }

                countdownDiv.textContent = '📸';
                await new Promise(resolve => setTimeout(resolve, 300));

                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', quality);

                stream.getTracks().forEach(track => track.stop());
                document.body.removeChild(div);

                return imageData;

            } catch(err) {
                const container = document.getElementById('camera-container');
                if (container && document.body.contains(container)) {
                    document.body.removeChild(container);
                }
                throw err;
            }
        }
        ''')

        display(js_code)
        time.sleep(1)

        try:
            data = eval_js(f'takePhoto(0.9, {countdown})')

            if not data:
                print("❌ Camera capture failed - no data received")
                return None

            binary = b64decode(data.split(',')[1])
            with open(save_path, 'wb') as f:
                f.write(binary)

            img = cv2.imread(save_path)
            if img is None:
                print("❌ Failed to read captured image")
                return None

            print(f"✅ Photo captured: {img.shape[1]}x{img.shape[0]} pixels")

            # Display captured image
            plt.figure(figsize=(6, 5))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('📸 Captured Photo (Full Frame)', fontsize=12, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

            return save_path

        except Exception as e:
            print(f"❌ Capture error: {e}")
            print("\n💡 Troubleshooting:")
            print("  1. Allow camera permissions when prompted")
            print("  2. Check no other app is using your camera")
            print("  3. Try refreshing the page and running again")
            return None

    def verify_with_webcam(self, document_path: str,
                          model: str = 'VGG-Face',
                          enhance_document: bool = True,
                          enhance_webcam: bool = True,
                          show_preprocessing: bool = True,
                          countdown: int = 3) -> dict:

        print("\n" + "="*70)
        print("🎥 ENHANCED KYC FACE VERIFICATION SYSTEM")
        print("="*70 + "\n")

        # Step 1: Extract and prepare document face
        print("📄 STEP 1: Processing document photo...")
        doc_result = self.face_processor.extract_face(document_path)

        if doc_result is None:
            print("❌ No face found in document")
            return {'success': False, 'match': False, 'error': 'No face in document'}

        doc_face_raw, doc_info = doc_result
        print(f"✅ Face extracted (confidence: {doc_info['confidence']:.2f})")
        print(f"   Face size: {doc_face_raw.shape[1]}x{doc_face_raw.shape[0]} pixels")

        # Save raw extracted face
        cv2.imwrite('/content/document_face_raw.jpg', doc_face_raw)

        # Enhance if requested
        if enhance_document:
            print("🔧 Enhancing document face quality...")
            doc_face = self.face_processor.enhance_face(doc_face_raw)
            doc_path = '/content/document_enhanced.jpg'
        else:
            doc_face = doc_face_raw
            doc_path = '/content/document_extracted.jpg'

        cv2.imwrite(doc_path, doc_face)

        # Step 2: Capture from webcam
        print(f"\n📸 STEP 2: Webcam capture")
        print("👤 Position your face clearly in front of the camera")
        print("💡 Ensure good lighting and similar pose to ID photo\n")

        webcam_path = self.capture_from_webcam(
            save_path='/content/webcam_capture.jpg',
            countdown=countdown
        )

        if not webcam_path:
            return {'success': False, 'match': False, 'error': 'Webcam capture failed'}

        # Step 3: Extract face from webcam capture
        print("\n🔍 STEP 3: Extracting face from webcam capture...")
        webcam_result = self.face_processor.extract_face(webcam_path)

        if webcam_result is None:
            print("❌ No face detected in webcam capture")
            return {'success': False, 'match': False, 'error': 'No face in webcam capture'}

        webcam_face_raw, webcam_info = webcam_result
        print(f"✅ Face detected (confidence: {webcam_info['confidence']:.2f})")
        print(f"   Face size: {webcam_face_raw.shape[1]}x{webcam_face_raw.shape[0]} pixels")

        # Save raw extracted face
        cv2.imwrite('/content/webcam_face_raw.jpg', webcam_face_raw)

        # Enhance webcam face for consistency
        if enhance_webcam:
            print("🔧 Enhancing webcam face quality...")
            webcam_face = self.face_processor.enhance_face(webcam_face_raw)
            webcam_final_path = '/content/webcam_face_enhanced.jpg'
        else:
            webcam_face = webcam_face_raw
            webcam_final_path = '/content/webcam_face_extracted.jpg'

        cv2.imwrite(webcam_final_path, webcam_face)

        # Step 3.5: Show preprocessing comparison
        if show_preprocessing:
            self._show_preprocessing(doc_face_raw, doc_face, webcam_face_raw, webcam_face,
                                    enhance_document, enhance_webcam)

        # Step 4: Face matching
        print(f"\n🔍 STEP 4: Comparing faces using {model}...")
        print(f"   Using: {'Enhanced' if enhance_document else 'Raw'} document face")
        print(f"   Using: {'Enhanced' if enhance_webcam else 'Raw'} webcam face")

        try:
            result = DeepFace.verify(
                img1_path=doc_path,
                img2_path=webcam_final_path,
                model_name=model,
                distance_metric='cosine',
                enforce_detection=False,
                align=True
            )

            distance = result['distance']
            threshold = result['threshold']
            confidence = max(0, min(100, (1 - distance / threshold) * 100))

            # CRITICAL FIX: Override match result if confidence is too low
            verified = result['verified']
            if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                verified = False
                print(f"\n⚠️  Confidence ({confidence:.1f}%) below minimum threshold ({self.MIN_CONFIDENCE_THRESHOLD}%)")
                print("   Overriding match result to NO MATCH")

            match_result = {
                'success': True,
                'match': verified,  # Use overridden value
                'confidence': confidence,
                'distance': distance,
                'threshold': threshold,
                'model': model,
                'doc_detection_conf': doc_info['confidence'],
                'webcam_detection_conf': webcam_info['confidence'],
                'timestamp': datetime.now().isoformat(),
                'preprocessing': {
                    'document_enhanced': enhance_document,
                    'webcam_enhanced': enhance_webcam
                },
                'low_confidence_override': confidence < self.MIN_CONFIDENCE_THRESHOLD
            }

        except Exception as e:
            print(f"❌ Face matching failed: {e}")
            return {'success': False, 'match': False, 'error': str(e)}

        # Step 5: Display results
        self._visualize_results(doc_path, webcam_final_path, match_result)
        self._print_results(match_result)

        return match_result

    def _show_preprocessing(self, doc_raw, doc_enhanced, webcam_raw, webcam_enhanced,
                           doc_is_enhanced, webcam_is_enhanced):
        """Show before/after preprocessing comparison."""
        print("\n🔍 PREPROCESSING COMPARISON:")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Document - Before
        axes[0, 0].imshow(cv2.cvtColor(doc_raw, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('📄 Document Face - EXTRACTED', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        # Document - After
        axes[0, 1].imshow(cv2.cvtColor(doc_enhanced, cv2.COLOR_BGR2RGB))
        title = '📄 Document Face - ENHANCED' if doc_is_enhanced else '📄 Document Face - USED'
        axes[0, 1].set_title(title, fontsize=11, fontweight='bold',
                            color='green' if doc_is_enhanced else 'blue')
        axes[0, 1].axis('off')

        # Webcam - Before
        axes[1, 0].imshow(cv2.cvtColor(webcam_raw, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('📸 Webcam Face - EXTRACTED', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')

        # Webcam - After
        axes[1, 1].imshow(cv2.cvtColor(webcam_enhanced, cv2.COLOR_BGR2RGB))
        title = '📸 Webcam Face - ENHANCED' if webcam_is_enhanced else '📸 Webcam Face - USED'
        axes[1, 1].set_title(title, fontsize=11, fontweight='bold',
                            color='green' if webcam_is_enhanced else 'blue')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
        print("✅ Both faces cropped to face-only (background removed)")
        print(f"✅ Document face: {'Enhanced' if doc_is_enhanced else 'Raw'}")
        print(f"✅ Webcam face: {'Enhanced' if webcam_is_enhanced else 'Raw'}\n")

    def _visualize_results(self, doc_path: str, webcam_path: str, result: dict):
        """Display side-by-side comparison with verdict."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Document face
        doc_img = cv2.imread(doc_path)
        axes[0].imshow(cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('📄 Document Face (Used for Matching)',
                         fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Webcam face
        webcam_img = cv2.imread(webcam_path)
        axes[1].imshow(cv2.cvtColor(webcam_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('📸 Webcam Face (Used for Matching)',
                         fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # Verdict - FIXED: Don't show MATCH for low confidence
        if result['match']:
            axes[2].text(0.5, 0.5, '✅\nMATCH',
                        ha='center', va='center',
                        fontsize=50, color='green', fontweight='bold')
            axes[2].set_title(f"Confidence: {result['confidence']:.1f}%",
                             fontsize=14, color='green', fontweight='bold')
        else:
            # Check if it was overridden due to low confidence
            if result.get('low_confidence_override', False):
                axes[2].text(0.5, 0.5, '❌\nNO MATCH',
                            ha='center', va='center',
                            fontsize=40, color='red', fontweight='bold')
                axes[2].text(0.5, 0.2, '⚠️ Low Confidence',
                            ha='center', va='center',
                            fontsize=14, color='orange', fontweight='bold')
                axes[2].set_title(f"Confidence: {result['confidence']:.1f}% (< {self.MIN_CONFIDENCE_THRESHOLD}%)",
                                 fontsize=12, color='red', fontweight='bold')
            else:
                axes[2].text(0.5, 0.5, '❌\nNO MATCH',
                            ha='center', va='center',
                            fontsize=50, color='red', fontweight='bold')
                axes[2].set_title(f"Confidence: {result['confidence']:.1f}%",
                                 fontsize=14, color='red', fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    def _print_results(self, result: dict):
        """Print formatted verification results."""
        print("\n" + "="*70)
        print("📊 VERIFICATION RESULTS")
        print("="*70)

        if result['match']:
            print("\n✅ ✅ ✅  IDENTITY VERIFIED  ✅ ✅ ✅")
        else:
            if result.get('low_confidence_override', False):
                print("\n❌ ❌ ❌  IDENTITY VERIFICATION FAILED (LOW CONFIDENCE)  ❌ ❌ ❌")
            else:
                print("\n❌ ❌ ❌  IDENTITY VERIFICATION FAILED  ❌ ❌ ❌")

        print(f"\nMatch Confidence: {result['confidence']:.1f}%")

        # Show low confidence warning
        if result.get('low_confidence_override', False):
            print(f"⚠️  WARNING: Confidence below {self.MIN_CONFIDENCE_THRESHOLD}% threshold - Match overridden to NO MATCH")

        print(f"Distance Score: {result['distance']:.4f}")
        print(f"Threshold: {result['threshold']:.4f}")
        print(f"Model Used: {result['model']}")

        print(f"\nDetection Quality:")
        print(f"  Document: {result['doc_detection_conf']:.1%}")
        print(f"  Webcam:   {result['webcam_detection_conf']:.1%}")

        print(f"\nPreprocessing Applied:")
        print(f"  Document: {'✅ Enhanced' if result['preprocessing']['document_enhanced'] else '❌ Raw'}")
        print(f"  Webcam:   {'✅ Enhanced' if result['preprocessing']['webcam_enhanced'] else '❌ Raw'}")

        # Interpretation
        if result['match']:
            if result['confidence'] > 70:
                status = "🎉 HIGH CONFIDENCE - Identity verified!"
            elif result['confidence'] > 50:
                status = "⚠️  MEDIUM CONFIDENCE - Manual review recommended"
            else:
                status = "⚠️  LOW CONFIDENCE - Additional verification required"
        else:
            if result.get('low_confidence_override', False):
                status = f"🚫 VERIFICATION FAILED - Confidence too low (< {self.MIN_CONFIDENCE_THRESHOLD}%)"
            else:
                status = "🚫 VERIFICATION FAILED - Identity does not match"

        print(f"\n{status}")
        print("="*70 + "\n")

# ============================================================================
# Cell 5: BANKING KYC PIPELINE
# ============================================================================

class BankingKYCPipeline:
    """Complete KYC verification pipeline with enhanced preprocessing options."""

    def __init__(self):
        self.face_processor = FaceProcessor()
        self.webcam_matcher = WebcamFaceMatcher(self.face_processor)
        print("✅ Banking KYC Pipeline initialized!")

    def verify_customer(self,
                       id_document_path: str,
                       model: str = 'VGG-Face',
                       enhance_document: bool = True,
                       enhance_webcam: bool = True,
                       show_preprocessing: bool = True,
                       countdown: int = 3) -> Dict:

        print("\n" + "="*70)
        print("🏦 ENHANCED BANKING KYC VERIFICATION PIPELINE")
        print("📸 MODE: Live Webcam Capture with Face-Only Comparison")
        print(f"⚠️  Minimum Confidence Threshold: {WebcamFaceMatcher.MIN_CONFIDENCE_THRESHOLD}%")
        print("="*70 + "\n")

        # Run enhanced verification
        result = self.webcam_matcher.verify_with_webcam(
            document_path=id_document_path,
            model=model,
            enhance_document=enhance_document,
            enhance_webcam=enhance_webcam,
            show_preprocessing=show_preprocessing,
            countdown=countdown
        )

        # Add risk assessment
        if result['success']:
            result['risk_score'] = self._calculate_risk_score(result)
            result['approved'] = self._make_decision(result)
            result['decision'] = self._get_decision_text(result)
        else:
            result['risk_score'] = 100
            result['approved'] = False
            result['decision'] = f"REJECTED - {result.get('error', 'Unknown error')}"

        return result

    def _calculate_risk_score(self, result: Dict) -> float:
        """Calculate risk score based on verification results."""
        risk = 0

        # Face match confidence (most important)
        if not result['match']:
            risk += 80
        else:
            risk += (100 - result['confidence']) * 0.5

        # Extra penalty for low confidence override
        if result.get('low_confidence_override', False):
            risk += 20

        # Detection quality
        doc_conf = result['doc_detection_conf']
        webcam_conf = result['webcam_detection_conf']

        if doc_conf < 0.95:
            risk += (1 - doc_conf) * 20

        if webcam_conf < 0.95:
            risk += (1 - webcam_conf) * 15

        return min(100, risk)

    def _make_decision(self, result: Dict) -> bool:
        """Make approval decision based on risk score."""
        # Auto-reject if low confidence override
        if result.get('low_confidence_override', False):
            return False
        return result['risk_score'] < 30

    def _get_decision_text(self, result: Dict) -> str:
        """Get human-readable decision text."""
        if result.get('low_confidence_override', False):
            return "REJECTED - Confidence Below Minimum Threshold"

        risk = result['risk_score']

        if risk < 30:
            return "APPROVED - Low Risk"
        elif risk < 60:
            return "MANUAL REVIEW - Medium Risk"
        else:
            return "REJECTED - High Risk"

    def generate_report(self, result: Dict) -> str:
        """Generate comprehensive verification report."""
        report = "\n" + "="*70 + "\n"
        report += "         BANKING KYC VERIFICATION REPORT\n"
        report += "="*70 + "\n\n"

        report += f"Timestamp: {result.get('timestamp', 'N/A')}\n"
        report += f"Capture Method: WEBCAM (LIVE) - FACE-ONLY COMPARISON\n"
        report += f"Min Confidence Threshold: {WebcamFaceMatcher.MIN_CONFIDENCE_THRESHOLD}%\n\n"

        report += "FINAL DECISION:\n"
        report += f"  Status: {result.get('decision', 'N/A')}\n"
        report += f"  Approved: {'YES ✅' if result.get('approved', False) else 'NO ❌'}\n"
        report += f"  Risk Score: {result.get('risk_score', 100):.1f}/100\n\n"

        report += "VERIFICATION DETAILS:\n" + "-"*70 + "\n"

        if result.get('success'):
            report += f"Face Match: {'✅ MATCH' if result['match'] else '❌ NO MATCH'}\n"

            if result.get('low_confidence_override', False):
                report += f"  ⚠️  Low Confidence Override Applied\n"

            report += f"  Confidence: {result['confidence']:.1f}%\n"
            report += f"  Distance: {result['distance']:.4f}\n"
            report += f"  Threshold: {result['threshold']:.4f}\n"
            report += f"  Model: {result['model']}\n\n"

            report += f"Detection Quality:\n"
            report += f"  Document: {result['doc_detection_conf']:.1%}\n"
            report += f"  Webcam: {result['webcam_detection_conf']:.1%}\n\n"

            if 'preprocessing' in result:
                report += f"Preprocessing:\n"
                report += f"  Document: {'Enhanced' if result['preprocessing']['document_enhanced'] else 'Raw'}\n"
                report += f"  Webcam: {'Enhanced' if result['preprocessing']['webcam_enhanced'] else 'Raw'}\n"
        else:
            report += f"Error: {result.get('error', 'Unknown error')}\n"

        report += "\n" + "="*70 + "\n"

        return report

    def save_result(self, result: Dict, filepath: str = '/content/kyc_result.json'):
        """Save verification result to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"✅ Result saved to {filepath}")


# ============================================================================
# TEST RUN
# ============================================================================

if __name__ == "__main__":
    kyc = BankingKYCPipeline()

    print("\n🎯 RUNNING ENHANCED VERIFICATION (Both faces enhanced)")
    print(f"⚠️  Matches with confidence < {WebcamFaceMatcher.MIN_CONFIDENCE_THRESHOLD}% will be rejected\n")

    result = kyc.verify_customer(
        id_document_path='',
        model='ArcFace',
        enhance_document=True,
        enhance_webcam=True,
        show_preprocessing=True,
        countdown=3
    )

    print(kyc.generate_report(result))

    # Save result
    kyc.save_result(result)

    # Final verdict
    if result.get('approved', False):
        print("\n🎉 ✅ KYC VERIFICATION PASSED - CUSTOMER APPROVED! ✅ 🎉\n")
    else:
        print(f"\n❌ KYC VERIFICATION FAILED - {result.get('decision', 'REJECTED')} ❌\n")




