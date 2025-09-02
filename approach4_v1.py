import spacy
import re
import json
from pathlib import Path
import fitz  # PyMuPDF for PDF parsing
from docx import Document  # python-docx for Word documents
import warnings
import traceback

class ModernCVParser:
    def __init__(self):
        warnings.filterwarnings("ignore")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Installing spaCy English model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_text_from_docx(self, docx_path):
        """Extract text from DOCX using python-docx"""
        try:
            doc = Document(docx_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_contact_info(self, text):
        """Extract contact information using regex - FIXED VERSION"""
        try:
            # Email extraction
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            
            # Phone extraction - FIXED PATTERN
            phone_patterns = [
                r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
                r'\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}',  # (123) 456-7890
                r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',      # 123-456-7890
                r'\d{10,}',                             # 1234567890
            ]
            
            phones = []
            for pattern in phone_patterns:
                matches = re.findall(pattern, text)
                phones.extend(matches)
            
            # Clean and filter phone numbers
            phones = [p.strip() for p in phones if len(p.strip()) >= 7]
            
            # LinkedIn extraction
            linkedin_pattern = r'linkedin\.com/in/[\w-]+'
            linkedin = re.findall(linkedin_pattern, text.lower())
            
            # GitHub extraction
            github_pattern = r'github\.com/[\w-]+'
            github = re.findall(github_pattern, text.lower())
            
            return {
                "emails": list(set(emails)),
                "phones": list(set(phones)),
                "linkedin": list(set(linkedin)),
                "github": list(set(github))
            }
        except Exception as e:
            print(f"Error in contact info extraction: {e}")
            return {"emails": [], "phones": [], "linkedin": [], "github": []}
    
    def extract_skills(self, text):
        """Extract skills using keyword matching"""
        try:
            skill_keywords = [
                'python', 'java', 'javascript', 'react', 'nodejs', 'sql', 'mysql',
                'postgresql', 'mongodb', 'docker', 'kubernetes', 'aws', 'azure',
                'git', 'html', 'css', 'angular', 'vue', 'django', 'flask',
                'machine learning', 'data analysis', 'excel', 'tableau', 'powerbi',
                'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'go', 'rust',
                'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
                'jenkins', 'ci/cd', 'agile', 'scrum', 'jira', 'confluence',
                'linux', 'windows', 'macos', 'bash', 'powershell'
            ]
            
            text_lower = text.lower()
            found_skills = []
            
            for skill in skill_keywords:
                if skill in text_lower:
                    found_skills.append(skill.title())
            
            return list(set(found_skills))
        except Exception as e:
            print(f"Error in skills extraction: {e}")
            return []
    
    def extract_education(self, text):
        """Extract education information"""
        try:
            degree_patterns = [
                r'\b(bachelor|master|phd|doctorate|diploma|certificate|b\.?\s*[a-z]+|m\.?\s*[a-z]+|ph\.?\s*d\.?)\b',
                r'\b(undergraduate|graduate|postgraduate)\b'
            ]
            
            degrees = []
            for pattern in degree_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                degrees.extend(matches)
            
            return list(set(degrees))
        except Exception as e:
            print(f"Error in education extraction: {e}")
            return []
    
    def extract_experience(self, text):
        """Extract work experience indicators"""
        try:
            exp_patterns = [
                r'(\d+)[\s\-\+]*year[s]?[\s]*(?:of\s)?(?:experience|exp)',
                r'(\d+)[\s\-\+]*yr[s]?[\s]*(?:of\s)?(?:experience|exp)',
                r'experience[:\s]*(\d+)[\s]*year[s]?',
                r'(\d+)[\s]*year[s]?[\s]*experience'
            ]
            
            experience = []
            for pattern in exp_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                experience.extend(matches)
            
            return list(set(experience))
        except Exception as e:
            print(f"Error in experience extraction: {e}")
            return []
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        try:
            # Limit text size for spaCy processing to avoid memory issues
            if len(text) > 10000:
                text = text[:10000]
                
            doc = self.nlp(text)
            
            persons = []
            organizations = []
            locations = []
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    persons.append(ent.text.strip())
                elif ent.label_ == "ORG":
                    organizations.append(ent.text.strip())
                elif ent.label_ in ["GPE", "LOC"]:
                    locations.append(ent.text.strip())
            
            return {
                "persons": list(set([p for p in persons if len(p) > 2])),
                "organizations": list(set([o for o in organizations if len(o) > 2])),
                "locations": list(set([l for l in locations if len(l) > 2]))
            }
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return {"persons": [], "organizations": [], "locations": []}
    
    def parse_cv(self, file_path):
        """Main parsing function that returns JSON-compatible dictionary"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                text = self.extract_text_from_docx(str(file_path))
            else:
                return {"error": f"Unsupported file format: {file_path.suffix}"}
            
            # Validate extracted text
            if not text or not isinstance(text, str):
                return {"error": "Failed to extract text from file"}
            
            if len(text.strip()) < 20:
                return {"error": "File appears to be empty or contains very little text"}
            
            print(f"Successfully extracted {len(text)} characters from {file_path.name}")
            
            # Extract all information with individual error handling
            contact_info = self.extract_contact_info(text)
            skills = self.extract_skills(text)
            education = self.extract_education(text)
            experience = self.extract_experience(text)
            entities = self.extract_entities(text)
            
            # Combine results in a structured JSON format
            result = {
                "file_info": {
                    "filename": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "text_length": len(text)
                },
                "contact_information": contact_info,
                "skills": skills,
                "education": education,
                "experience_years": experience,
                "named_entities": entities,
                "extraction_metadata": {
                    "extraction_successful": True,
                    "parser_version": "modern_cv_parser_v1.1_fixed"
                }
            }
            
            return result
            
        except Exception as e:
            # Print full traceback for debugging
            error_details = traceback.format_exc()
            print(f"Full error traceback:\n{error_details}")
            
            return {
                "error": f"Failed to parse CV: {str(e)}",
                "error_type": type(e).__name__,
                "extraction_metadata": {
                    "extraction_successful": False,
                    "parser_version": "modern_cv_parser_v1.1_fixed"
                }
            }
    
    def parse_cv_to_json(self, file_path, output_path=None):
        """Parse CV and return/save as JSON string"""
        result = self.parse_cv(file_path)
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"Results saved to {output_path}")
        
        return json_output

# Usage example
def main():
    parser = ModernCVParser()
    
    # Replace with your CV file path
    cv_file = "F:/Cogntix/Unblit/AI/SampleCVs/Sample1.pdf"  
    
    try:
        result_json = parser.parse_cv_to_json(cv_file, "parsed_cv.json")
        print("CV parsed successfully!")
        print(result_json)
        
    except Exception as e:
        print(f"Error in main: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
