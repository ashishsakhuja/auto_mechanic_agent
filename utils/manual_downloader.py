import requests
import tempfile
import os
import zipfile
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

def download_file(url, suffix=''):
    """Download a URL to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        tmp_file.write(response.content)
        return tmp_file.name

def extract_pdf_text(pdf_path):
    """Extract plain text from every page of a PDF."""
    text = ''
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_html_text(html_path, keywords=None):
    """
    Read an HTML file and either:
      - return the full visible text (if keywords is None), or
      - pull out only those <h1>-<h5> sections whose heading contains one of your keywords.
    """
    with open(html_path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    if not keywords:
        # dump all visible text
        return soup.get_text(separator="\n", strip=True)

    kws = [k.lower() for k in keywords]
    sections = []
    # find all headings
    for heading in soup.find_all(["h1","h2","h3","h4","h5"]):
        ht = heading.get_text(strip=True).lower()
        if any(kw in ht for kw in kws):
            block = [heading.get_text(strip=True)]
            # collect siblings until the next same-level heading
            for sib in heading.find_next_siblings():
                if sib.name and sib.name.startswith("h") and len(sib.name) == len(heading.name):
                    break
                block.append(sib.get_text(separator="\n", strip=True))
            sections.append("\n".join(block))

    return "\n\n".join(sections)

def unzip_and_extract_manuals(zip_path, keywords=None):
    """
    Unzip a bundle, then for each PDF or HTML file inside it, extract text.
    Returns a list of (relative_filename, extracted_text).
    """
    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmp_dir)

        for root, _, files in os.walk(tmp_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, tmp_dir)
                ext = fn.lower().rsplit('.', 1)[-1]
                if ext == 'pdf':
                    results.append((rel, extract_pdf_text(full)))
                elif ext in ('html','htm'):
                    results.append((rel, extract_html_text(full, keywords)))
    return results

def fetch_and_parse_manual(bundle_url, keywords=None):
    """
    Fetch a PDF or ZIP from bundle_url, then parse it:
      - PDFs → extract all text
      - ZIPs → unzip and extract PDFs + HTML (optionally filtering HTML by keywords)
    Returns a single combined string.
    """
    # 1) PDF case
    if bundle_url.lower().endswith('.pdf'):
        pdf_path = download_file(bundle_url, '.pdf')
        try:
            return extract_pdf_text(pdf_path)
        finally:
            os.remove(pdf_path)

    # 2) ZIP case
    elif bundle_url.lower().endswith('.zip'):
        zip_path = download_file(bundle_url, '.zip')
        try:
            extracted = unzip_and_extract_manuals(zip_path, keywords=keywords or ["oil","filter","drain"])
            combined = []
            for fname, txt in extracted:
                combined.append(f"\n--- {fname} ---\n{txt}")
            return "\n".join(combined)
        finally:
            os.remove(zip_path)

    # 3) neither
    else:
        return "ERROR: Unsupported file type for URL: " + bundle_url
