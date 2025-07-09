# knowledge/vehicle_knowledge_source.py

from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class ManualIndex:
    """
    Builds an index of all PDFs in a given folder (and subfolders) by filename.
    """

    def __init__(self, manuals_dir: str, index_dir: str = "manual_index"):
        # 1) Resolve path relative to this file
        project_root = Path(__file__).resolve().parent.parent
        manual_path = Path(manuals_dir)
        if not manual_path.exists():
            manual_path = project_root / manuals_dir

        # 2) Find all PDF files, recursively
        pdf_files = list(manual_path.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found under {manual_path!r}. "
                "Make sure your Toyota manuals are placed there."
            )

        # 3) Build one tiny doc per PDF (using the filename as content)
        docs = [
            Document(page_content=pdf.stem, metadata={"path": str(pdf)})
            for pdf in pdf_files
        ]

        # 4) Embed those filename-docs into a small Chroma store
        embeddings = OpenAIEmbeddings()
        self.pdf_index = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=index_dir
        )

    def find_manuals(self, query: str, k: int = 1):
        """
        Return up to k Documents whose metadata['path'] points to
        the best-matching PDF(s) for the given query.
        """
        retriever = self.pdf_index.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)
