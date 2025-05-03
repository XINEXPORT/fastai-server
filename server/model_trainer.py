import pymupdf 
from PIL import Image
import pytesseract
import io
from PyPDF2 import PdfReader
import re

def extract_text_from_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image)

def preprocess_text(text):
    # Remove unwanted characters, extra spaces, etc.
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_financial_pages(pdf_path, start_page=10, end_page=16):
    reader = PdfReader(pdf_path)
    pages = reader.pages[start_page-1:end_page]
    text = ''
    for page in pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return preprocess_text(text)

def train_model_with_pdf(pdf_filename):

    doc = pymupdf.open(pdf_filename)
    image_info_list = []

    for page_num in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_num)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_info_list.append({
                "page": page_num + 1,
                "image_bytes": base_image["image"],
                "ext": base_image["ext"]
            })

    print(f"✅ Extracted {len(image_info_list)} images from '{pdf_filename}'")

    for image_info in image_info_list:
        image = Image.open(io.BytesIO(image_info["image_bytes"]))
        text = pytesseract.image_to_string(image)
        image_info["ocr_text"] = text

    financial_text = extract_financial_pages(pdf_filename)
    print("financial text", len(financial_text), financial_text[:10])

    return financial_text
