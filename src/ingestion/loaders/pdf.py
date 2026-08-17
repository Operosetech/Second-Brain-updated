import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF locally using pypdf.
    Falls back to pdfplumber for pages that yield no text (e.g. image-heavy pages).
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"PDF has {total_pages} pages.")
            # 1. Create an empty list with exactly enough slots for every page
            text_parts: list[str] = [""] * total_pages
            blank_pages: list[int] = []

           # 2. Pass 1 (pypdf): Insert text into the correct index
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_parts[i] = text  # Put text in its exact page slot
                else:
                    blank_pages.append(i) # Save the index (0-based) for later

           # 3. Pass 2 (pdfplumber): Fill in the missing slots
            if blank_pages:
                logfire.info(f"pypdf returned blank on pages {blank_pages} — retrying.")
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        for page_idx in blank_pages:
                            page = pdf.pages[page_idx]
                            fallback_text = page.extract_text() or ""
                            if fallback_text.strip():
                                text_parts[page_idx] = fallback_text # Insert at the correct slot
                except Exception as plumber_err:
                    logfire.warning(f"pdfplumber fallback failed: {plumber_err}")

            # 4. Filter out any slots that are still empty, then join in perfect order
            clean_parts = [text for text in text_parts if text.strip()]
            full_text = "\n".join(clean_parts)

            if not full_text.strip():
                logfire.warning(f"No text extracted from {file_path}. File may be fully image-based.")
            else:
                logfire.info(f"Extracted {len(full_text)} characters from {file_path}.")

            return full_text

        except Exception as e:
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")
            raise

if __name__ == "__main__":
    pdf_directory = "/path/to/your/pdfs"
    result = parse_pdf(pdf_directory)
    print(result)