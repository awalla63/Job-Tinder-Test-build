"""Extract plain text from uploaded resumes.

Library choices are deliberate:

* PDF -- pdfplumber (MIT) first, pypdf (BSD) as a fallback when pdfplumber returns
  little or nothing. PyMuPDF/fitz is faster and handles columns better, but it is
  AGPL-3.0: its network-use clause would reach a hosted web app serving these parses,
  so it is not worth the licensing exposure here.

* DOCX -- docx2python, not python-docx. `python-docx`'s `document.paragraphs` silently
  skips table cells, headers/footers and text boxes. Resume templates routinely put the
  entire skills sidebar in a table or a text box, so a naive python-docx parse drops
  exactly the content the matcher most needs, and does it without raising anything.
"""
from __future__ import annotations

import io
import logging
import re

log = logging.getLogger(__name__)

MAX_CHARS = 20000
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")


class UnsupportedResume(ValueError):
    pass


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKLINE_RE.sub("\n\n", text)
    return text.strip()[:MAX_CHARS]


def _from_pdf(payload: bytes) -> str:
    text = ""
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(payload)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        log.warning("pdfplumber failed, falling back to pypdf: %s", exc)

    # A near-empty result usually means a scanned or unusually laid-out PDF; pypdf
    # sometimes recovers text pdfplumber misses, so it is worth the second attempt.
    if len(text.strip()) < 100:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(payload))
            fallback = "\n".join(page.extract_text() or "" for page in reader.pages)
            if len(fallback.strip()) > len(text.strip()):
                text = fallback
        except Exception as exc:
            log.warning("pypdf fallback failed: %s", exc)

    return text


def _from_docx(payload: bytes) -> str:
    from docx2python import docx2python

    with docx2python(io.BytesIO(payload)) as doc:
        return doc.text


def extract_text(payload: bytes, filename: str) -> str:
    """Return the plain text of an uploaded resume.

    Raises UnsupportedResume for unknown types or when nothing usable comes out --
    storing an empty resume would produce a meaningless embedding and an empty deck,
    which is far more confusing to the user than an upload error.
    """
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        text = _from_pdf(payload)
    elif name.endswith(".docx"):
        text = _from_docx(payload)
    elif name.endswith(".txt"):
        text = payload.decode("utf-8", errors="replace")
    elif name.endswith(".doc"):
        raise UnsupportedResume(
            "Legacy .doc files are not supported. Please save as .docx or PDF."
        )
    else:
        raise UnsupportedResume("Upload a PDF, DOCX or TXT resume.")

    text = _tidy(text)
    if len(text) < 100:
        raise UnsupportedResume(
            "Could not read enough text from that file. If it is a scanned PDF, "
            "please upload a text-based PDF or a DOCX instead."
        )
    return text
