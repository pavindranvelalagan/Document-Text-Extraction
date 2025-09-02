# parse_cv_docling.py
# Parse a CV PDF with Docling and output both:
# 1) docling_raw.json  - the full Docling document model
# 2) structured_cv.json - clean, meaningful fields for CV analysis

import json
import re
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PipelineOptions

# -------- Settings --------
DEFAULT_OUTPUT_DIR = "docling_cv_output"

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

MONTH_TOKENS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

def parse_contact(line: str) -> dict:
    out = {}
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', line)
    phone = re.findall(r'[\+()]?[\d\s\-]{8,}', line)
    linkedin = re.findall(r'(?:linkedin\.com/in/[\w\-]+)', line, flags=re.I)
    github = re.findall(r'(?:github\.com/[\w\-]+)', line, flags=re.I)
    if email: out["email"] = email
    if phone:
        cleaned = re.sub(r"[^\d\+]", "", phone)
        if len(cleaned) >= 10:
            out["phone"] = cleaned
    if linkedin:
        url = linkedin
        if not url.startswith("http"):
            url = "https://" + url
        out["linkedin"] = url
    if github:
        url = github
        if not url.startswith("http"):
            url = "https://" + url
        out["github"] = url
    return out

def likely_name(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 60:
        return False
    words = t.split()
    if not (2 <= len(words) <= 5):
        return False
    if any(k in t.lower() for k in ["experience","education","skills","projects","summary","referees","references"]):
        return False
    return all(re.match(r"^[A-Za-z'.\-]+$", w) for w in words)

def is_date_like(text: str) -> bool:
    t = text.lower()
    if re.search(r'\b(19|20)\d{2}\b', t): return True
    if any(m in t for m in MONTH_TOKENS): return True
    return False

def split_skills(items):
    skills = []
    for item in items:
        for part in re.split(r"[,\|/•;]+", item):
            s = part.strip()
            if 2 <= len(s) <= 40 and re.match(r"^[A-Za-z0-9\.\+#\-\s]+$", s):
                skills.append(s)
    return sorted(set(skills))

def deref(ref, texts_by_ref, groups_by_ref):
    key = ref.get("$ref","")
    if key.startswith("#/texts/"):
        return texts_by_ref.get(key)
    if key.startswith("#/groups/"):
        return groups_by_ref.get(key)
    return None

def parse_docling_json(doc: dict) -> dict:
    texts = {t["self_ref"]: t for t in doc.get("texts", [])}
    groups = {g["self_ref"]: g for g in doc.get("groups", [])}
    body = doc["body"]

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
            has_content = any(current_job.get(k) for k in ["title","company","dates"]) or current_job.get("bullets")
            if has_content:
                current_job["bullets"] = current_job.get("bullets", [])
                result["experience"].append(current_job)
        current_job = None

    def flush_edu():
        nonlocal current_edu
        if current_edu:
            has_content = any(current_edu.get(k) for k in ["degree","institution","dates"]) or current_edu.get("details")
            if has_content:
                current_edu["details"] = current_edu.get("details", [])
                result["education"].append(current_edu)
        current_edu = None

    for child_ref in body.get("children", []):
        node = deref(child_ref, texts, groups)
        if not node:
            continue

        label = node.get("label")
        name = node.get("name")
        text = node.get("text","").strip()

        if label == "section_header":
            header = text

            if not saw_name and likely_name(header):
                result["name"] = header
                saw_name = True
                current_section = None
                continue

            canonical = TOP_SECTIONS_MAP.get(header.upper())
            if canonical:
                flush_job()
                flush_edu()
                current_section = canonical
                continue

            if current_section == "experience":
                flush_job()
                current_job = {"title": header, "company": "", "dates": "", "bullets": []}
                continue

            if current_section == "education":
                flush_edu()
                current_edu = {"degree": header, "institution": "", "dates": "", "details": []}
                continue

        elif label == "text":
            if saw_name and not result["contact"]:
                c = parse_contact(text)
                if c: result["contact"] = c

            if current_section == "summary":
                result["summary"] = (result["summary"] + " " + text).strip()
            elif current_section == "experience":
                if current_job:
                    if is_date_like(text):
                        current_job["dates"] = text
                    elif not current_job.get("company") and len(text.split()) <= 10:
                        current_job["company"] = text
            elif current_section == "education":
                if current_edu:
                    if is_date_like(text):
                        current_edu["dates"] = text
                    elif not current_edu.get("institution"):
                        current_edu["institution"] = text
                    else:
                        current_edu["details"].append(text)
            elif current_section == "projects":
                if text: result["projects"].append({"name":"", "description": text})
            elif current_section == "certifications":
                if text: result["certifications"].append(text)
            elif current_section == "references":
                if text: result["references"].append(text)

        elif name == "list":
            items = []
            for li_ref in node.get("children", []):
                li = deref(li_ref, texts, groups)
                if li and li.get("label") == "list_item":
                    items.append(li.get("text","").strip())
            if current_section == "experience" and current_job:
                current_job["bullets"].extend(items)
            elif current_section == "education" and current_edu:
                current_edu["details"].extend(items)
            elif current_section == "skills":
                result["skills"].extend(split_skills(items))

    flush_job()
    flush_edu()
    result["skills"] = sorted(set([s for s in result["skills"] if s and len(s) <= 40]))
    return result

def parse_with_docling(pdf_path: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Enable OCR to be robust for scanned/image-heavy PDFs (optional)
    pipeline_options = PipelineOptions()
    pipeline_options.do_ocr = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: pipeline_options}
    )

    # Convert PDF -> DoclingDocument
    result = converter.convert(pdf_path)

    # Export full Docling dict (authoritative JSON view)
    doc_dict = result.document.export_to_dict()  # use dict for JSON export
    (out_dir / "docling_raw.json").write_text(json.dumps(doc_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build structured CV JSON from the Docling dict
    structured = parse_docling_json(doc_dict)
    (out_dir / "structured_cv.json").write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional human QA: markdown view
    md = result.document.export_to_markdown()
    (out_dir / "cv_markdown.md").write_text(md, encoding="utf-8")

    return structured

import argparse

def main():
    print("🚀 Docling CV Parser")
    print("====================")
    parser = argparse.ArgumentParser(description="Parse a CV PDF with Docling and output JSON.")
    parser.add_argument("pdf_path", help="Path to CV PDF")
    parser.add_argument("output_dir", nargs="?", default="docling_cv_output", help="Output folder")
    args = parser.parse_args()

    structured = parse_with_docling(args.pdf_path, args.output_dir)
    print(f"\n✅ Done. Results in: {args.output_dir}")
    print(f"   - docling_raw.json")
    print(f"   - structured_cv.json")
    print(f"   - cv_markdown.md")
    print(f"\nDetected name: {structured.get('name','')}")
    print(f"Experience entries: {len(structured.get('experience',[]))}")
    print(f"Education entries: {len(structured.get('education',[]))}")
    print(f"Skills count: {len(structured.get('skills',[]))}")


if __name__ == "__main__":
    main()
