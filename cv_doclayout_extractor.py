# cv_semantic_parser_fixed.py - Simplified Semantic Parser
import os
import pdfplumber
from transformers import pipeline
import pandas as pd
import json
import re
from collections import defaultdict

class SimplifiedSemanticCVParser:
    def __init__(self):
        """Initialize with semantic classification models only"""
        
        print("Initializing Simplified Semantic CV Parser")
        print("=" * 60)
        
        # Load semantic classification models
        self.section_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        self.ner_model = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            grouped_entities=True
        )
        
        # Define resume sections for classification
        self.resume_sections = [
            "Contact Information",
            "Personal Details", 
            "Professional Summary",
            "Work Experience",
            "Education", 
            "Skills",
            "Technical Skills",
            "Projects",
            "Certifications",
            "References",
            "Awards",
            "Languages"
        ]
        
        print("Loaded semantic classification models")
    
    def extract_text_blocks(self, pdf_path):
        """Extract text blocks using simple paragraph-based splitting"""
        
        print(f"Extracting text blocks from PDF...")
        
        text_blocks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract all text from page
                    page_text = page.extract_text() or ""
                    
                    if not page_text.strip():
                        continue
                    
                    # Split into meaningful chunks based on double newlines and headings
                    # This simulates finding text regions
                    chunks = []
                    
                    # Split by double newlines first
                    paragraphs = page_text.split('\n\n')
                    
                    for para in paragraphs:
                        if len(para.strip()) > 20:  # Only include substantial text
                            chunks.append(para.strip())
                    
                    # Also split by single newlines for shorter sections
                    if not chunks:  # Fallback if no double newlines
                        lines = page_text.split('\n')
                        current_chunk = ""
                        
                        for line in lines:
                            line = line.strip()
                            if len(line) > 5:
                                if len(current_chunk) > 100:  # Start new chunk
                                    chunks.append(current_chunk)
                                    current_chunk = line
                                else:
                                    current_chunk += " " + line
                        
                        if current_chunk:
                            chunks.append(current_chunk)
                    
                    # Add chunks to text blocks
                    for chunk in chunks:
                        if len(chunk.strip()) > 15:  # Filter very short chunks
                            text_blocks.append({
                                'page': page_num,
                                'text': chunk,
                                'block_id': len(text_blocks)
                            })
                            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return []
        
        print(f"Extracted {len(text_blocks)} text blocks")
        return text_blocks
    
    def classify_text_section(self, text):
        """Classify what type of resume section this text belongs to"""
        
        if len(text.strip()) < 10:
            return "Other", 0.0
        
        try:
            # Use zero-shot classification
            result = self.section_classifier(text, candidate_labels=self.resume_sections)
            
            section = result["labels"][0]
            confidence = result["scores"][0]
            
            return section, confidence
            
        except Exception as e:
            print(f"Error classifying text: {e}")
            return "Other", 0.0
    
    def extract_entities(self, text):
        """Extract named entities"""
        
        try:
            entities = self.ner_model(text)
            
            entity_dict = defaultdict(list)
            for entity in entities:
                entity_type = entity['entity_group']
                entity_text = entity['word'].replace('##', '')  # Clean subword tokens
                if entity_text not in entity_dict[entity_type]:
                    entity_dict[entity_type].append(entity_text)
            
            return dict(entity_dict)
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {}
    
    def extract_contact_info(self, text):
        """Extract contact information using regex"""
        
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['emails'] = list(set(emails))
        
        # Phone pattern (enhanced)
        phone_pattern = r'[\+]?[0-9\s\-\(\)]{10,}'
        phones = re.findall(phone_pattern, text)
        if phones:
            # Clean phone numbers
            cleaned_phones = []
            for phone in phones:
                cleaned = re.sub(r'[^\d\+]', '', phone)
                if len(cleaned) >= 10:
                    cleaned_phones.append(cleaned)
            if cleaned_phones:
                contact_info['phones'] = list(set(cleaned_phones))
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text.lower())
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
        
        return contact_info
    
    def process_cv_semantic(self, pdf_path):
        """Main processing pipeline"""
        
        print(f"\nProcessing CV: {os.path.basename(pdf_path)}")
        print("=" * 60)
        
        # Step 1: Extract text blocks
        text_blocks = self.extract_text_blocks(pdf_path)
        
        if not text_blocks:
            print("No text blocks found")
            return None
        
        # Step 2: Classify each block
        semantic_data = defaultdict(list)
        all_contact_info = {}
        
        for block in text_blocks:
            text = block['text']
            
            # Skip very short text
            if len(text.strip()) < 10:
                continue
            
            print(f"Processing block {block['block_id']}: {text[:50]}...")
            
            # Classify section type
            section_type, confidence = self.classify_text_section(text)
            
            # Extract entities
            entities = self.extract_entities(text)
            
            # Extract contact info
            contact_info = self.extract_contact_info(text)
            if contact_info:
                all_contact_info.update(contact_info)
            
            block_data = {
                'block_id': block['block_id'],
                'text': text,
                'section_type': section_type,
                'confidence': confidence,
                'entities': entities,
                'char_count': len(text)
            }
            
            semantic_data[section_type].append(block_data)
            
            print(f"   → {section_type} (confidence: {confidence:.2f})")
        
        # Step 3: Structure final output
        structured_cv = self.create_structured_output(semantic_data, all_contact_info)
        
        return structured_cv
    
    def create_structured_output(self, semantic_data, all_contact_info):
        """Create final structured CV output"""
        
        structured = {
            'name': '',
            'contact_details': all_contact_info,
            'professional_summary': '',
            'education': [],
            'work_experience': [],
            'skills': [],
            'technical_skills': [],
            'projects': [],
            'certifications': [],
            'languages': [],
            'awards': [],
            'references': []
        }
        
        # Extract name from entities
        all_names = []
        for section_blocks in semantic_data.values():
            for block in section_blocks:
                if 'PER' in block['entities']:
                    all_names.extend(block['entities']['PER'])
        
        if all_names:
            # Take the first name found
            structured['name'] = all_names[0]
        
        # Process each semantic section
        for section_type, blocks in semantic_data.items():
            combined_text = '\n\n'.join([block['text'] for block in blocks])
            
            if section_type == 'Professional Summary':
                structured['professional_summary'] = combined_text
                
            elif section_type == 'Work Experience':
                structured['work_experience'].append({
                    'content': combined_text,
                    'entities': [block['entities'] for block in blocks]
                })
                
            elif section_type == 'Education':
                structured['education'].append({
                    'content': combined_text,
                    'entities': [block['entities'] for block in blocks]
                })
                
            elif section_type in ['Skills', 'Technical Skills']:
                # Extract individual skills
                skills_text = combined_text.lower()
                # Common skill separators
                skills = re.split(r'[,\n\|•\-]+', skills_text)
                cleaned_skills = []
                for skill in skills:
                    skill = skill.strip()
                    if len(skill) > 2 and len(skill) < 50:  # Reasonable skill length
                        cleaned_skills.append(skill.title())
                
                if section_type == 'Technical Skills':
                    structured['technical_skills'].extend(cleaned_skills)
                else:
                    structured['skills'].extend(cleaned_skills)
                
            elif section_type == 'Projects':
                structured['projects'].append({
                    'content': combined_text,
                    'entities': [block['entities'] for block in blocks]
                })
                
            elif section_type == 'Certifications':
                structured['certifications'].append({
                    'content': combined_text,
                    'entities': [block['entities'] for block in blocks]
                })
        
        # Remove duplicates from skills
        structured['skills'] = list(set(structured['skills']))
        structured['technical_skills'] = list(set(structured['technical_skills']))
        
        return structured
    
    def save_results(self, structured_data, output_dir):
        """Save structured results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        json_path = os.path.join(output_dir, 'structured_cv.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        # Save readable summary
        summary_path = os.path.join(output_dir, 'cv_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("STRUCTURED CV EXTRACTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Name: {structured_data.get('name', 'Not detected')}\n\n")
            
            f.write("Contact Details:\n")
            contact = structured_data.get('contact_details', {})
            f.write(f"   Email: {contact.get('emails', 'Not found')}\n")
            f.write(f"   Phone: {contact.get('phones', 'Not found')}\n")
            f.write(f"   LinkedIn: {contact.get('linkedin', 'Not found')}\n\n")
            
            f.write("Professional Summary:\n")
            summary = structured_data.get('professional_summary', 'Not found')
            f.write(f"{summary[:300]}{'...' if len(summary) > 300 else ''}\n\n")
            
            f.write("Skills:\n")
            all_skills = structured_data.get('skills', []) + structured_data.get('technical_skills', [])
            for skill in all_skills[:20]:  # Show first 20 skills
                f.write(f"   • {skill}\n")
            if len(all_skills) > 20:
                f.write(f"   • ... and {len(all_skills) - 20} more\n")
            
            f.write(f"\nWork Experience: {len(structured_data.get('work_experience', []))} sections found\n")
            f.write(f"Education: {len(structured_data.get('education', []))} sections found\n")
            f.write(f"Projects: {len(structured_data.get('projects', []))} sections found\n")
            f.write(f"Certifications: {len(structured_data.get('certifications', []))} sections found\n")
        
        print(f"Saved structured data: {json_path}")
        print(f"Saved summary: {summary_path}")
        
        return json_path, summary_path

def main():
    """Main function"""
    
    print("SIMPLIFIED SEMANTIC CV PARSER")
    print("=" * 60)
    print("Direct Text Extraction + Semantic Classification")
    print("=" * 60)
    
    # Get PDF path
    pdf_path = input("Enter path to your CV PDF: ").strip()
    if not pdf_path:
        pdf_path = "F:\\Cogntix\\Unblit\\AI\\SampleCVs\\Sample1.pdf"
        print(f"Using default: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
    
    try:
        # Initialize parser
        parser = SimplifiedSemanticCVParser()
        
        # Process CV
        structured_data = parser.process_cv_semantic(pdf_path)
        
        if structured_data:
            # Save results
            json_path, summary_path = parser.save_results(structured_data, "semantic_cv_output")
            
            print(f"\nSUCCESS! Extracted structured CV data:")
            print(f"Results saved to: semantic_cv_output/")
            print(f"\nKey Information:")
            print(f"   Name: {structured_data.get('name', 'Not detected')}")
            print(f"   Email: {structured_data.get('contact_details', {}).get('emails', 'Not found')}")
            print(f"   Phone: {structured_data.get('contact_details', {}).get('phones', 'Not found')}")
            print(f"   Skills: {len(structured_data.get('skills', []) + structured_data.get('technical_skills', []))}")
            print(f"   Work Experience: {len(structured_data.get('work_experience', []))}")
            print(f"   Education: {len(structured_data.get('education', []))}")
            print(f"   Projects: {len(structured_data.get('projects', []))}")
            
            print(f"\nSample JSON output:")
            sample = {k: v for k, v in list(structured_data.items())[:3]}
            print(json.dumps(sample, indent=2)[:400] + "...")
            
            print(f"\nThe JSON file can now be fed directly to your LLM for analysis!")
            
        else:
            print("Failed to extract structured data")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
