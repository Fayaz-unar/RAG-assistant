from pypdf import PdfReader


def load_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_number
            })

    return pages
