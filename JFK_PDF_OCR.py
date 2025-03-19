#!/usr/bin/env python
import os
import io
import argparse
import logging
import hashlib
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
import cv2
import numpy as np
from tqdm import tqdm

# For custom binarization methods
from skimage.filters import threshold_sauvola, threshold_niblack

def get_pdf_links(url):
    """Scrape PDF links from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.lower().endswith('.pdf'):
                if not href.startswith('http'):
                    href = requests.compat.urljoin(url, href)
                pdf_links.append(href)
        return pdf_links
    except Exception as e:
        logging.error(f"Error getting PDF links from {url}: {e}")
        return []

def download_pdf(pdf_url, pdf_dir):
    """
    Download a PDF from the URL if not present locally.
    Returns the PDF content (bytes) and the local file path.
    """
    file_name = pdf_url.split("/")[-1]
    file_path = os.path.join(pdf_dir, file_name)
    if os.path.exists(file_path):
        logging.info(f"File exists locally: {file_path}")
        with open(file_path, "rb") as f:
            return f.read(), file_path
    else:
        try:
            logging.info(f"Downloading PDF: {pdf_url}")
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_bytes = response.content
            with open(file_path, "wb") as f:
                f.write(pdf_bytes)
            return pdf_bytes, file_path
        except Exception as e:
            logging.error(f"Failed to download {pdf_url}: {e}")
            return None, file_path

def compute_md5(file_bytes):
    """Compute MD5 hash for given bytes."""
    m = hashlib.md5()
    m.update(file_bytes)
    return m.hexdigest()

def extract_text_pypdf2(pdf_bytes):
    """Extract text from PDF using PyPDF2."""
    text = ""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logging.error(f"PyPDF2 extraction error: {e}")
    return text.strip()

def extract_text_pdfplumber(pdf_bytes):
    """Extract text from PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logging.error(f"pdfplumber extraction error: {e}")
    return text.strip()

def deskew_image(image):
    """
    Detects and corrects the skew of a grayscale image.
    Uses the minimum-area rectangle method to determine the angle.
    """
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image  # Return original if no text is found
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def ocr_with_config(image, config):
    """
    Apply a single pre-processing pipeline based on the given config,
    then run OCR on the resulting image.
    
    Config dictionary keys:
      - median_blur: int (kernel size for median blur)
      - contrast_enhance: bool (apply CLAHE)
      - deskew: bool (apply deskewing)
      - morph_op: str (one of "erode", "dilate", "open", "close", or None)
      - morph_kernel: int (kernel size for morphological op)
      - threshold_method: str ("otsu", "adaptive", "sauvola", or "niblack")
      - adaptive_block_size: int (for adaptive thresholding)
      - adaptive_C: int (for adaptive thresholding)
      - sauvola_window: int (for Sauvola thresholding)
      - niblack_window: int (for Niblack thresholding)
      - niblack_k: float (for Niblack thresholding)
    """
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if config.get("contrast_enhance", False):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    
    if config.get("deskew", False):
        gray = deskew_image(gray)
    
    median_blur = config.get("median_blur", 3)
    blurred = cv2.medianBlur(gray, median_blur)
    
    morph_op = config.get("morph_op", None)
    if morph_op:
        kernel_size = config.get("morph_kernel", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if morph_op == "erode":
            blurred = cv2.erode(blurred, kernel, iterations=1)
        elif morph_op == "dilate":
            blurred = cv2.dilate(blurred, kernel, iterations=1)
        elif morph_op == "open":
            blurred = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        elif morph_op == "close":
            blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    threshold_method = config.get("threshold_method", "otsu")
    if threshold_method == "adaptive":
        block_size = config.get("adaptive_block_size", 11)
        C = config.get("adaptive_C", 2)
        processed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, C)
    elif threshold_method == "sauvola":
        window_size = config.get("sauvola_window", 25)
        thresh_sauvola = threshold_sauvola(blurred, window_size=window_size)
        processed = (blurred > thresh_sauvola).astype("uint8") * 255
    elif threshold_method == "niblack":
        window_size = config.get("niblack_window", 25)
        k = config.get("niblack_k", 0.8)
        thresh_niblack = threshold_niblack(blurred, window_size=window_size, k=k)
        processed = (blurred > thresh_niblack).astype("uint8") * 255
    else:
        _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    text = pytesseract.image_to_string(processed, lang='eng', config='--oem 1 --psm 3')
    return text, processed

def ensemble_ocr_on_image(image, configs):
    """
    Apply multiple pre-processing configurations (ensemble OCR) on the same image,
    compare OCR outputs using pairwise similarity, and select the best result.
    """
    ocr_results = []
    for config in configs:
        text, _ = ocr_with_config(image, config)
        ocr_results.append(text)
    n = len(ocr_results)
    if n == 0:
        return ""
    best_text = None
    best_score = -1
    for i in range(n):
        total_sim = 0
        count = 0
        for j in range(n):
            if i != j:
                sim = difflib.SequenceMatcher(None, ocr_results[i], ocr_results[j]).ratio()
                total_sim += sim
                count += 1
        avg_sim = total_sim / count if count > 0 else 0
        if avg_sim > best_score:
            best_score = avg_sim
            best_text = ocr_results[i]
    return best_text

def create_composite_image(images, spacing=10, background_color=255):
    """
    Create a composite image by centering each image horizontally and stacking them vertically.
    """
    widths = [img.shape[1] for img in images]
    max_width = max(widths)
    padded_images = []
    for img in images:
        h, w = img.shape[:2]
        pad_left = (max_width - w) // 2
        pad_right = max_width - w - pad_left
        padded = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=background_color)
        padded_images.append(padded)
    spacing_img = np.full((spacing, max_width), background_color, dtype=np.uint8)
    composite = padded_images[0]
    for img in padded_images[1:]:
        composite = np.vstack((composite, spacing_img, img))
    return composite

def ocr_pdf(pdf_bytes, poppler_path=None, processed_images_dir=None, composite_dir=None,
            composite_pdf_dir=None, pdf_name=None, use_ensemble=False, configs=None):
    """
    Extract text from a PDF using OCR.
    
    If use_ensemble is True, multiple pre-processing pipelines are run on each page,
    and a consensus result is chosen.
    If processed_images_dir and pdf_name are provided, a sample processed image is saved per page.
    If composite_dir is provided, a composite image for each page (of multiple pipelines) is saved.
    If composite_pdf_dir is provided, a single composite image of all pages (stacked vertically) is created.
    """
    ocr_text = ""
    pdf_page_images = []  # To store sample processed image for each page
    try:
        if poppler_path:
            images = convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
        else:
            images = convert_from_bytes(pdf_bytes)
        for page_idx, image in enumerate(images):
            processed_images = []
            if use_ensemble and configs:
                texts = []
                for config in configs:
                    text, proc_img = ocr_with_config(image, config)
                    texts.append(text)
                    processed_images.append(proc_img)
                text = ensemble_ocr_on_image(image, configs)
            else:
                text, proc_img = ocr_with_config(image, {"median_blur": 3, "threshold_method": "otsu"})
                processed_images.append(proc_img)
            # Save sample processed image for the page if requested
            if processed_images_dir and pdf_name:
                os.makedirs(processed_images_dir, exist_ok=True)
                processed_filename = f"{pdf_name}_page_{page_idx+1}.png"
                processed_filepath = os.path.join(processed_images_dir, processed_filename)
                if not cv2.imwrite(processed_filepath, processed_images[0]):
                    logging.error(f"Failed to save processed image to: {processed_filepath}")
                else:
                    logging.info(f"Processed image saved to: {processed_filepath}")
            # Save composite image of pipelines for the page if requested
            if composite_dir and pdf_name and len(processed_images) > 1:
                os.makedirs(composite_dir, exist_ok=True)
                composite = create_composite_image(processed_images, spacing=10)
                composite_filename = f"{pdf_name}_page_{page_idx+1}_composite.png"
                composite_filepath = os.path.join(composite_dir, composite_filename)
                if not cv2.imwrite(composite_filepath, composite):
                    logging.error(f"Failed to save composite image to: {composite_filepath}")
                else:
                    logging.info(f"Composite image saved to: {composite_filepath}")
            # Save sample processed image for composite PDF generation
            pdf_page_images.append(processed_images[0])
            ocr_text += text + "\n"
        # After processing all pages, create a single composite image stacking all pages if requested
        if composite_pdf_dir and pdf_name and pdf_page_images:
            os.makedirs(composite_pdf_dir, exist_ok=True)
            pdf_composite = create_composite_image(pdf_page_images, spacing=20)
            composite_pdf_filename = f"{pdf_name}_pages_composite.png"
            composite_pdf_filepath = os.path.join(composite_pdf_dir, composite_pdf_filename)
            if not cv2.imwrite(composite_pdf_filepath, pdf_composite):
                logging.error(f"Failed to save PDF composite image to: {composite_pdf_filepath}")
            else:
                logging.info(f"PDF composite image saved to: {composite_pdf_filepath}")
    except Exception as e:
        logging.error(f"OCR extraction failed: {e}")
    return ocr_text.strip()

def process_pdf(pdf_url, pdf_dir, text_dir, poppler_path=None, processed_images_dir=None,
                composite_dir=None, composite_pdf_dir=None, use_ensemble=False, configs=None, force_ocr=False):
    """
    Process a single PDF:
      1. Download (or load from cache)
      2. Attempt extraction with PyPDF2/pdfplumber unless force_ocr is True,
         then use OCR (with ensemble if enabled).
      3. Save extracted text to a local file.
      4. Optionally create a composite image of all pages.
    Returns the extracted text.
    """
    logging.info(f"Processing PDF: {pdf_url}")
    pdf_bytes, local_file_path = download_pdf(pdf_url, pdf_dir)
    if not pdf_bytes:
        return None

    text = ""
    if not force_ocr:
        text = extract_text_pypdf2(pdf_bytes)
        if text:
            logging.info("Text successfully extracted using PyPDF2.")
        else:
            logging.info("No text from PyPDF2; trying pdfplumber.")
            text = extract_text_pdfplumber(pdf_bytes)
    if force_ocr or not text:
        logging.info("Falling back to OCR processing.")
        file_name = pdf_url.split("/")[-1]
        pdf_name = os.path.splitext(file_name)[0]
        text = ocr_pdf(pdf_bytes, poppler_path=poppler_path,
                       processed_images_dir=processed_images_dir,
                       composite_dir=composite_dir,
                       composite_pdf_dir=composite_pdf_dir,
                       pdf_name=pdf_name,
                       use_ensemble=use_ensemble, configs=configs)
    if not text:
        logging.warning("Failed to extract any text from the PDF.")
        return None

    file_name = pdf_url.split("/")[-1]
    text_file_path = os.path.join(text_dir, file_name + ".txt")
    try:
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Extracted text saved to: {text_file_path}")
    except Exception as e:
        logging.error(f"Failed to save text for {pdf_url}: {e}")
    return text

def process_and_save(pdf_url, pdf_dir, text_dir, poppler_path=None, processed_images_dir=None,
                     composite_dir=None, composite_pdf_dir=None, use_ensemble=False, configs=None, force_ocr=False):
    """Wrapper to process a PDF and print a sample output."""
    text = process_pdf(pdf_url, pdf_dir, text_dir, poppler_path, processed_images_dir,
                       composite_dir, composite_pdf_dir, use_ensemble, configs, force_ocr)
    if text:
        sample = text[:300]
        logging.info(f"Sample OCR output for {pdf_url}:\n{sample}\n")
    else:
        logging.info(f"Skipping {pdf_url} as no text was extracted.")
    return pdf_url

def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF Processing with Advanced Pre‑processing, Ensemble OCR, and PDF Composite Images")
    parser.add_argument("--url", default="https://www.archives.gov/research/jfk/release-2025", help="URL containing PDF links")
    parser.add_argument("--pdf-dir", default="pdf_files", help="Directory to store downloaded PDFs")
    parser.add_argument("--text-dir", default="pdf_contents", help="Directory to store extracted text")
    parser.add_argument("--poppler-path", default=None, help="Path to Poppler binaries if not in PATH")
    parser.add_argument("--num-workers", type=int, default=5, help="Number of concurrent workers")
    parser.add_argument("--save-processed", action="store_true", help="Save a sample processed image for each page")
    parser.add_argument("--processed-dir", default="processed_images", help="Directory to store sample processed images")
    parser.add_argument("--save-all-pipeline", action="store_true", help="Save composite images of all pre-processing pipelines per page")
    parser.add_argument("--composite-dir", default="composite_images", help="Directory to store composite images of page pipelines")
    # New flags for creating a composite image for all pages of the PDF
    parser.add_argument("--save-pdf-composite", action="store_true", help="Save a composite image of all pages from the PDF")
    parser.add_argument("--pdf-composite-dir", default="pdf_composites", help="Directory to store composite images of all PDF pages")
    # Advanced pre-processing options
    parser.add_argument("--median-blur", type=int, default=3, help="Median blur kernel size")
    parser.add_argument("--adaptive-block-size", type=int, default=11, help="Adaptive threshold block size")
    parser.add_argument("--adaptive-C", type=int, default=2, help="Adaptive threshold constant C")
    parser.add_argument("--sauvola-window", type=int, default=25, help="Window size for Sauvola thresholding")
    parser.add_argument("--niblack-window", type=int, default=25, help="Window size for Niblack thresholding")
    parser.add_argument("--niblack-k", type=float, default=0.8, help="k value for Niblack thresholding")
    parser.add_argument("--contrast-enhance", action="store_true", help="Enable contrast enhancement using CLAHE")
    parser.add_argument("--deskew", action="store_true", help="Enable deskewing of the image")
    parser.add_argument("--morph-op", choices=["erode", "dilate", "open", "close"], help="Apply a morphological operation")
    parser.add_argument("--morph-kernel", type=int, default=3, help="Kernel size for the morphological operation")
    parser.add_argument("--threshold-method", choices=["otsu", "adaptive", "sauvola", "niblack"], default="otsu", help="Binarization method")
    parser.add_argument("--use-ensemble", action="store_true", help="Use ensemble OCR with multiple pre-processing configurations")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR processing even if text is extractable via other methods")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    os.makedirs(args.pdf_dir, exist_ok=True)
    os.makedirs(args.text_dir, exist_ok=True)
    if args.save_processed:
        os.makedirs(args.processed_dir, exist_ok=True)
    if args.save_all_pipeline:
        os.makedirs(args.composite_dir, exist_ok=True)
    if args.save_pdf_composite:
        os.makedirs(args.pdf_composite_dir, exist_ok=True)

    logging.info("Starting PDF processing...")

    configs = [
        {"median_blur": args.median_blur, "threshold_method": args.threshold_method,
         "contrast_enhance": args.contrast_enhance, "deskew": args.deskew,
         "morph_op": args.morph_op, "morph_kernel": args.morph_kernel},
        {"median_blur": args.median_blur + 2, "threshold_method": args.threshold_method,
         "contrast_enhance": args.contrast_enhance, "deskew": args.deskew,
         "morph_op": args.morph_op, "morph_kernel": args.morph_kernel},
        {"median_blur": args.median_blur, "threshold_method": "adaptive",
         "adaptive_block_size": args.adaptive_block_size, "adaptive_C": args.adaptive_C,
         "contrast_enhance": args.contrast_enhance, "deskew": args.deskew,
         "morph_op": args.morph_op, "morph_kernel": args.morph_kernel},
        {"median_blur": args.median_blur, "threshold_method": "sauvola",
         "sauvola_window": args.sauvola_window,
         "contrast_enhance": args.contrast_enhance, "deskew": args.deskew,
         "morph_op": args.morph_op, "morph_kernel": args.morph_kernel},
        {"median_blur": args.median_blur, "threshold_method": "niblack",
         "niblack_window": args.niblack_window, "niblack_k": args.niblack_k,
         "contrast_enhance": args.contrast_enhance, "deskew": args.deskew,
         "morph_op": args.morph_op, "morph_kernel": args.morph_kernel},
    ] if args.use_ensemble else None

    pdf_links = get_pdf_links(args.url)
    logging.info(f"Found {len(pdf_links)} PDF files.")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_and_save, pdf_url, args.pdf_dir, args.text_dir,
                            args.poppler_path,
                            args.processed_dir if args.save_processed else None,
                            args.composite_dir if args.save_all_pipeline else None,
                            args.pdf_composite_dir if args.save_pdf_composite else None,
                            args.use_ensemble, configs, args.force_ocr): pdf_url
            for pdf_url in pdf_links
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing a PDF: {e}")

    logging.info("PDF processing complete.")

if __name__ == "__main__":
    main()

