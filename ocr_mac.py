import fitz
import sys
import CoreFoundation
import Quartz
import Vision

def ocr_image(image_bytes):
    # Convert image bytes to CGImage
    data = CoreFoundation.CFDataCreate(None, image_bytes, len(image_bytes))
    source = Quartz.CGImageSourceCreateWithData(data, None)
    cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    
    recognized_text = []

    def handle_request(request, error):
        if error:
            print(f"Error handling request: {error}")
            return
        results = request.results()
        if not results:
            return
        for observation in results:
            top_candidate = observation.topCandidates_(1)[0]
            recognized_text.append(top_candidate.string())

    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handle_request)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    request.setRecognitionLanguages_(["es-ES", "en-US"])

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    success, error = handler.performRequests_error_([request], None)
    if not success:
        print(f"Failed to perform request: {error}")
    
    return "\n".join(recognized_text)

files = [
    "04A Newton Cotes.pdf",
    "04B lectura 1 Integracion_Numerica_presición_y_Error.pdf",
    "04C lectua 2 Simpson.pdf"
]

for fname in files:
    try:
        print(f"Processing {fname}...")
        doc = fitz.open(fname)
        full_text = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            text = ocr_image(img_bytes)
            full_text.append(f"--- Page {i + 1} ---\n{text}")
        
        out_name = fname + ".ocr.txt"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
        print(f"Successfully processed {fname}")
    except Exception as e:
        print(f"Failed {fname}: {e}")
