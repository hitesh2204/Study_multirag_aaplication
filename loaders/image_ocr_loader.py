# loaders/image_ocr_loader.py
from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)