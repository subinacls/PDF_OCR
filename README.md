python JFK_PDF_OCR.py \
    --url "https://www.archives.gov/research/jfk/release-2025" \
    --poppler-path "/path/to/poppler" \
    --use-ensemble \
    --median-blur 3 \
    --adaptive-block-size 11 \
    --adaptive-C 2 \
    --contrast-enhance \
    --deskew \
    --morph-op open \
    --morph-kernel 3 \
    --threshold-method otsu \
    --save-processed \
    --processed-dir "processed_images" \
    --save-all-pipeline \
    --composite-dir "composite_images" \
    --save-pdf-composite \
    --pdf-composite-dir "pdf_composites" \
    --force-ocr
