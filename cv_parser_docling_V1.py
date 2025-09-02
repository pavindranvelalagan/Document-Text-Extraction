# cv_parser_docling.py
# End-to-end CV extraction using Docling (no paid APIs).
# Requires: pip install docling

import argparse
import json
import re
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PipelineOptions

# --------------------------
# Configuration
# --------------------------

TOP_SECTIONS_MAP = {
    "SUMMARY": "summary",
    "WORK EXPERIENCE": "experience",
    "EXPERIENCE": "experience",
    "EDUCATION": "education",
    "KEY SKILLS": "skills",
    "SKILLS": "skills",
    "PROJECTS": "projects",
    "CERTIFICATES": "certifications",
    "CERTIFICATIONS": "certifications",
    "REFEREES": "references",
    "REFERENCES": "references",
}

# Common month tokens to detect date lines
MONTH_TOKENS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec"
]


# --------------------------
# Helpers
# --------------------------

def _parse_contact_from_text(line: str) -> dict:
    contact = {}
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', line)
    phone = re.findall(r'[\+()]?[\d\s\-]{8,}', line)
    linkedin = re.findall(r'(?:linkedin\.com/in/[\w\-]+)', line, flags=re.I)
    github = re.findall(r'(?:github\.com/[\w\-]+)', line, flags=re.I)

    if email:
        contact["email"] = email
    if phone:
        cleaned = re.sub(r"[^\d\+]", "", phone)
        if len(cleaned) >= 10:
            contact["phone"] = cleaned
    if linkedin:
        url = linkedin
        if not url.startswith("http"):
            url = "https://" + url
        contact["linkedin"] = url
    if github:
        url = github
        if not url.startswith("http"):
            url = "https://" + url
        contact["github"] = url

    return contact


def _likely_name(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 60:
        return False
    words = t.split()
    if not (2 <= len(words) <= 5):
        return False
    # Avoid obvious non-name headers
    if any(k in t.lower() for k in ["experience", "education", "skills", "projects", "summary", "referees", "references"]):
        return False
    # Mostly letters and punctuation common in names
    return all(re.match(r"^[A-Za-z'.\-]+$", w) for w in words)


def _is_date_like(text: str) -> bool:
    t = text.lower()
    if re.search(r'\b(19|20)\d{2}\b', t):
        return True
    if any(m in t for m in MONTH_TOKENS):
        return True
    return False


def _split_skills_from_bullets(items):
    skills = []
    for item in items:
        # Split by common delimiters
        parts = re.split(r"[,\|/•;]+", item)
        for p in parts:
            s = p.strip()
            if 2 <= len(s) <= 40:
                # Keep simple alnum/tech tokens, avoid sentences
                if re.match(r"^[A-Za-z0-9\.\+#\-\s]+$", s):
                    skills.append(s)
    # Deduplicate and sort
    return sorted(set(skills))


def _deref(ref, texts_by_ref, groups_by_ref):
    """Resolve a $ref entry to its node dict."""
    if "$ref" not in ref:
        return None
    key = ref["$ref"]
    if key.startswith("#/texts/"):
        return texts_by_ref.get(key)
    if key.startswith("#/groups/"):
        return groups_by_ref.get(key)
    return None


# --------------------------
# Core Parsing Logic
# --------------------------

def parse_docling_document(doc_dict: dict) -> dict:
    """
    Convert Docling's JSON (export_to_dict) into a structured CV JSON.
    """
    texts = {t["self_ref"]: t for t in doc_dict.get("texts", [])}
    groups = {g["self_ref"]: g for g in doc_dict.get("groups", [])}
    body = doc_dict["body"]

    result = {
        "name": "",
        "contact": {},
        "summary": "",
        "experience": [],
        "education": [],
        "skills": [],
        "projects": [],
        "certifications": [],
        "references": []
    }

    current_section = None
    saw_name = False
    current_job = None
    current_edu = None

    def flush_job():
        nonlocal current_job
        if current_job:
            # Keep only non-empty entries
            has_content = any(current_job.get(k) for k in ["title", "company", "dates"]) or current_job.get("bullets")
            if has_content:
                current_job["bullets"] = current_job.get("bullets", [])
                result["experience"].append(current_job)
        current_job = None

    def flush_edu():
        nonlocal current_edu
        if current_edu:
            has_content = any(current_edu.get(k) for k in ["degree", "institution", "dates"]) or current_edu.get("details")
            if has_content:
                current_edu["details"] = current_edu.get("details", [])
                result["education"].append(current_edu)
        current_edu = None

    # Iterate the document in reading order
    for child_ref in body["children"]:
        node = _deref(child_ref, texts, groups)
        if node is None:
            continue

        label = node.get("label")
        name = node.get("name")
        text = node.get("text", "").strip()

        # Section headers
        if label == "section_header":
            header = text

            # First header typically is the name
            if not saw_name and _likely_name(header):
                result["name"] = header
                saw_name = True
                current_section = None
                continue

            # Top sections
            canonical = TOP_SECTIONS_MAP.get(header.upper())
            if canonical:
                flush_job()
                flush_edu()
                current_section = canonical
                continue

            # Inside EXPERIENCE, subheaders are job titles
            if current_section == "experience":
                flush_job()
                current_job = {"title": header, "company": "", "dates": "", "bullets": []}
                continue

            # Inside EDUCATION, subheaders are degree lines
            if current_section == "education":
                flush_edu()
                current_edu = {"degree": header, "institution": "", "dates": "", "details": []}
                continue

        # Plain text lines
        elif label == "text":
            # Contact line often comes after name
            if saw_name and not result["contact"]:
                c = _parse_contact_from_text(text)
                if c:
                    result["contact"] = c

            if current_section == "summary":
                result["summary"] = (result["summary"] + " " + text).strip()

            elif current_section == "experience":
                if current_job:
                    if _is_date_like(text):
                        current_job["dates"] = text
                    elif not current_job.get("company"):
                        # Heuristic: short line likely company or location
                        if len(text.split()) <= 10:
                            current_job["company"] = text

            elif current_section == "education":
                if current_edu:
                    if _is_date_like(text):
                        current_edu["dates"] = text
                    elif not current_edu.get("institution"):
                        current_edu["institution"] = text
                    else:
                        current_edu["details"].append(text)

            elif current_section == "projects":
                if text:
                    result["projects"].append({"name": "", "description": text})

            elif current_section == "certifications":
                if text:
                    result["certifications"].append(text)

            elif current_section == "references":
                if text:
                    result["references"].append(text)

        # Lists of bullets (e.g., responsibilities, skills)
        elif name == "list":
            items = []
            for li_ref in node.get("children", []):
                li_node = _deref(li_ref, texts, groups)
                if li_node and li_node.get("label") == "list_item":
                    items.append(li_node.get("text", "").strip())

            if current_section == "experience":
                if current_job:
                    current_job["bullets"].extend(items)

            elif current_section == "education":
                if current_edu:
                    current_edu["details"].extend(items)

            elif current_section == "skills":
                result["skills"].extend(_split_skills_from_bullets(items))

    # Flush pending aggregates
    flush_job()
    flush_edu()

    # Final cleanup
    result["skills"] = sorted(set([s for s in result["skills"] if s and len(s) <= 40]))
    return result


# --------------------------
# Conversion + Orchestration
# --------------------------

def parse_cv_with_docling(pdf_path: str, output_dir: str = "docling_cv_output") -> dict:
    """
    Convert PDF to DoclingDocument, export to dict, parse structured CV.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline (enable OCR if you expect scanned PDFs)
    pipeline_options = PipelineOptions()
    pipeline_options.do_ocr = True  # set False if you don't need OCR

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: pipeline_options
        }
    )

    # 1) Convert
    result = converter.convert(pdf_path)

    # 2) Export Docling document
    doc_dict = result.document.export_to_dict()  # Docling-recommended for JSON export
    json_path = out_dir / "docling_raw.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)

    # 3) Parse to structured CV JSON
    structured = parse_docling_document(doc_dict)

    # 4) Save structured outputs
    structured_path = out_dir / "structured_cv.json"
    with structured_path.open("w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    # 5) Save Markdown (optional, for human QA)
    md = result.document.export_to_markdown()
    md_path = out_dir / "cv_markdown.md"
    md_path.write_text(md, encoding="utf-8")

    # 6) Write a short summary file
    summary_txt = out_dir / "cv_summary.txt"
    lines = []
    lines.append(f"Name: {structured.get('name','')}")
    c = structured.get("contact", {})
    lines.append(f"Email: {c.get('email','')}")
    lines.append(f"Phone: {c.get('phone','')}")
    lines.append("")
    lines.append(f"Summary: {structured.get('summary','')[:600]}")
    lines.append("")
    lines.append(f"Experience entries: {len(structured.get('experience',[]))}")
    lines.append(f"Education entries: {len(structured.get('education',[]))}")
    lines.append(f"Skills count: {len(structured.get('skills',[]))}")
    lines.append(f"Projects count: {len(structured.get('projects',[]))}")
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    return structured


# --------------------------
# CLI entry
# --------------------------

def main():
    print("Docling CV Parser")
    print("====================")

    parser = argparse.ArgumentParser(description="Parse CV PDF with Docling into structured JSON.")
    parser.add_argument("pdf_path", help="Path to CV PDF")
    parser.add_argument("output_dir", nargs="?", default="docling_cv_output", help="Output folder")
    args = parser.parse_args()

    structured = parse_cv_with_docling(args.pdf_path, args.output_dir)
    print(f"\nParsed. Structured JSON saved in: {args.output_dir}/structured_cv.json")
    print(f"   Detected name: {structured.get('name','')}")
    print(f"   Experience entries: {len(structured.get('experience',[]))}")
    print(f"   Education entries: {len(structured.get('education',[]))}")
    print(f"   Skills: {len(structured.get('skills',[]))}")

    if __name__ == "__main__":
        main()