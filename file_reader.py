"""
file_reader.py
----------------
Helper functions to extract plain text out of uploaded CSV and PDF files.
This powers the "/upload-file" bonus endpoint in main.py.
"""

import io
import pandas as pd
from pypdf import PdfReader


def read_csv_as_text(file_bytes: bytes) -> str:
    """
    Reads a CSV file (given as raw bytes) and converts it into a
    human-readable plain text representation.

    Args:
        file_bytes: Raw bytes of the uploaded .csv file.

    Returns:
        A string representation of the CSV content.

    Raises:
        ValueError: if the bytes cannot be parsed as a valid CSV.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not parse CSV file: {e}")

    if df.empty:
        return "(The uploaded CSV file is empty.)"

    # to_string() gives a clean, readable plain-text table
    return df.to_string(index=False)


def read_pdf_as_text(file_bytes: bytes) -> str:
    """
    Reads a PDF file (given as raw bytes) and extracts its plain text content.

    Args:
        file_bytes: Raw bytes of the uploaded .pdf file.

    Returns:
        The extracted text, with pages separated by a marker line.

    Raises:
        ValueError: if the bytes cannot be parsed as a valid PDF, or if no
                    extractable text is found (e.g. a scanned/image-only PDF).
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not parse PDF file: {e}")

    pages_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_text.append(f"--- Page {page_number} ---\n{text.strip()}")

    full_text = "\n\n".join(pages_text).strip()

    if not full_text:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image-only document."
        )

    return full_text
