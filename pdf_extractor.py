import fitz
# fitz is the import name for PyMuPDF
# PyMuPDF is the most reliable PDF text extraction library for Python

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF file given its raw bytes.

    file_bytes : raw bytes of the PDF file received from the upload
    Returns    : extracted text as a single string
    Raises     : ValueError if PDF contains no extractable text
                 (scanned image PDFs)
    """
    # Open PDF from bytes — fitz.open can accept bytes directly
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

    full_text = ""

    for page_num in range(len(pdf_document)):
        # Iterate through every page
        page = pdf_document[page_num]
        page_text = page.get_text()
        # get_text() extracts all readable text from a page
        # Returns empty string for image-only pages
        full_text += page_text + "\n"

    pdf_document.close()

    # Check if any text was actually extracted
    if len(full_text.strip()) < 50:
        # Less than 50 characters means either empty or scanned PDF
        raise ValueError(
            "No readable text found in this PDF. "
            "This may be a scanned image PDF. "
            "Please upload a text-selectable PDF."
        )

    return full_text

def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extract text from a plain .txt file.

    file_bytes : raw bytes of the text file
    Returns    : decoded text string
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Some files use different encoding
        return file_bytes.decode("latin-1")