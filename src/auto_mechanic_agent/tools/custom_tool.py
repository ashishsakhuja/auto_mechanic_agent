import os
import re
import uuid
import requests
from io import BytesIO
from typing import Type, Optional
import duckdb
from typing import Dict, List
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from openai import OpenAI

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    ListFlowable, ListItem, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# Determine the tests directory relative to this file
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')
os.makedirs(TESTS_DIR, exist_ok=True)

# ─────────────────────────── Image Generation Tool ──────────────────────────

_client = OpenAI()

class ImageGenInput(BaseModel):
    prompt: str = Field(..., description="A text prompt to generate your image")
    size: Optional[str] = Field("512x512", description="Image size, e.g. 256x256 or 512x512")

class ImageGenTool(BaseTool):
    name: str = "generate_image"
    description: str = "Generate an image from a text prompt via OpenAI and return the local file path."
    args_schema: Type[BaseModel] = ImageGenInput

    def _run(self, prompt: str, size: str = "512x512") -> str:
        resp = _client.images.generate(prompt=prompt, size=size, n=1)
        url = resp.data[0].url
        img_bytes = requests.get(url).content

        filename = f"generated_image_{uuid.uuid4().hex}.png"
        out_path = os.path.join(TESTS_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        return out_path


# ───────────────────────────── PDF Creation Tool ────────────────────────────

class PDFCreatorInput(BaseModel):
    html: str = Field(..., description="HTML (including <img> tags) to render")
    output_path: Optional[str] = Field(
        None,
        description="Where to write the PDF; defaults to the tests folder"
    )

class PDFCreatorTool(BaseTool):
    name: str = "pdf_creator"
    description: str = "Render HTML (h1,h2,p,ol,ul,img) into a PDF via ReportLab."
    args_schema: Type[BaseModel] = PDFCreatorInput

    def _run(self, html: str, output_path: Optional[str] = None) -> str:
        # --- Resolve output_path ---
        if output_path:
            output_path = os.path.expanduser(output_path)
        else:
            filename = f"solution_{uuid.uuid4().hex}.pdf"
            output_path = os.path.join(TESTS_DIR, filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 1) Convert any Markdown-style images into <img> tags
        html = re.sub(
            r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)',
            r'<img src="\g<src>" alt="\g<alt>"/>',
            html
        )
        # 2) Strip out any <br> tags
        html = re.sub(r'<br\s*/?>', ' ', html, flags=re.IGNORECASE)

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=40, leftMargin=40,
            topMargin=60, bottomMargin=40,
        )
        flowables = []

        # Embed each <img>; if it's a URL, download it first
        img_re = re.compile(
            r'<img\s+[^>]*?'
            r'src=[\'\"](?P<src>[^\'\"]+)[\'\"]'
            r'(?:[^>]*?width=[\'\"]?(?P<width>\d+)[\'\"]?)?'
            r'(?:[^>]*?height=[\'\"]?(?P<height>\d+)[\'\"]?)?'
            r'[^>]*?>',
            re.IGNORECASE
        )

        def _embed(match):
            src = match.group("src")
            w = match.group("width")
            h = match.group("height")

            # If it's a remote URL, fetch it into a BytesIO
            if src.lower().startswith(("http://", "https://")):
                try:
                    resp = requests.get(src)
                    resp.raise_for_status()
                    img_obj = BytesIO(resp.content)
                except Exception:
                    return ""  # skip this image
            else:
                # Otherwise treat as a local path
                src = os.path.expanduser(src)
                src = os.path.normpath(src)
                if not os.path.isabs(src):
                    src = os.path.abspath(src)
                img_obj = src

            try:
                img = RLImage(
                    img_obj,
                    width=int(w) if w else None,
                    height=int(h) if h else None
                )
                flowables.append(img)
                flowables.append(Spacer(1, 12))
            except Exception:
                pass

            return ""

        html = img_re.sub(_embed, html)

        # headings
        for h1 in re.findall(r"<h1>(.*?)</h1>", html, re.DOTALL | re.IGNORECASE):
            flowables.append(Paragraph(h1.strip(), styles["Heading1"]))
            flowables.append(Spacer(1, 12))
        for h2 in re.findall(r"<h2>(.*?)</h2>", html, re.DOTALL | re.IGNORECASE):
            flowables.append(Paragraph(h2.strip(), styles["Heading2"]))
            flowables.append(Spacer(1, 12))

        # paragraphs
        for p in re.findall(r"<p>(.*?)</p>", html, re.DOTALL | re.IGNORECASE):
            flowables.append(Paragraph(p.strip(), styles["Normal"]))
            flowables.append(Spacer(1, 12))

        # lists
        def extract_list(tag, bullet):
            pattern = rf"<{tag}>(.*?)</{tag}>"
            for block in re.findall(pattern, html, re.DOTALL | re.IGNORECASE):
                items = re.findall(r"<li>(.*?)</li>", block, re.DOTALL | re.IGNORECASE)
                lf = ListFlowable(
                    [ListItem(Paragraph(it.strip(), styles["Normal"])) for it in items],
                    bulletType=bullet
                )
                flowables.append(lf)
                flowables.append(Spacer(1, 12))

        extract_list("ol", "1")
        extract_list("ul", "bullet")

        doc.build(flowables)
        return os.path.abspath(output_path)

# ──────────────────────────────── Query Manifest Tool ───────────────────────────────────────

HERE      = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]  # .../src/auto_mechanic_agent/tools → up to repo root
DB_PATH   = REPO_ROOT / "knowledge" / "manuals.duckdb"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Couldn’t find DuckDB at {DB_PATH!r}")


class QueryArgs(BaseModel):
    make: str = Field(..., description="The vehicle make, e.g. Toyota")
    model: str = Field(..., description="The vehicle model or a substring thereof, e.g. Camry")
    year: str = Field(..., description="The model year, e.g. 2006")

class QueryManifestTool(BaseTool):
    name: str = "query_manifest"
    description: str = (
        "Generate and run a SQL query against the DuckDB `manifest` table to "
        "find the bundle_url for a given make/model/year. "
        "Columns: make TEXT, model TEXT, year TEXT, bundle_url TEXT."
    )
    args_schema: Type[QueryArgs] = QueryArgs

    def _run(self, make: str, model: str, year: str) -> List[Dict]:
        # connect to the absolute path
        conn = duckdb.connect(str(DB_PATH))
        sql = """
        SELECT bundle_url
          FROM manifest
         WHERE make = ?
           AND model ILIKE '%' || ? || '%'
           AND year = ?
         LIMIT 1;
        """
        df = conn.execute(sql, [make, model, year]).fetchdf()
        # return as a list of dicts
        return df.to_dict(orient="records")