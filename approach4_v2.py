import spacy
import re
import json
from pathlib import Path
import fitz
from docx import Document
import warnings

class ImprovedCVParser:
    def __init__(self):
        warnings.filterwarnings("ignore")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def extract_name(self, text):
        """Extract name from the first few lines"""
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            # Look for name patterns (usually first line, all caps or title case)
            if len(line) > 5 and len(line) < 50 and not '@' in line and not '+' in line:
                # Check if it looks like a name (contains only letters and spaces)
                if re.match(r'^[A-Za-z\s]+$', line):
                    return line
        return ""
    
    def extract_contact_info(self, text):
        """Extract contact information with better patterns"""
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Phone extraction - multiple patterns
        phone_patterns = [
            r'\+\d{1,3}\s?\(\d{1,2}\)\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}',  # +94(0) 76 97 87 765
            r'\+\d{1,3}\s?\d{2}\s?\d{7,10}',  # +94 11 2650301
            r'\+\d{1,3}\s?\d{2}\s?\d{3}\s?\d{4}',  # +94 71 2207030
            r'\(\d{3}\)\s?\d{3}[-\s]?\d{4}',  # (123) 456-7890
            r'\d{3}[-\s]?\d{3}[-\s]?\d{4}',   # 123-456-7890
        ]
        
        phones = []
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        # LinkedIn extraction
        linkedin_pattern = r'(?:linkedin\.com/in/[\w-]+|LinkedIn profile)'
        linkedin_matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
        
        return {
            "emails": list(set(emails)),
            "phones": list(set(phones)),
            "linkedin": linkedin_matches,
            "address": self.extract_address(text)
        }
    
    def extract_address(self, text):
        """Extract address information"""
        # Look for address-like patterns near contact info
        lines = text.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            if any(keyword in line.lower() for keyword in ['contact details:', 'address:']):
                # Extract everything after email and phone
                parts = line.split(',')
                address_parts = []
                for part in parts:
                    part = part.strip()
                    if '@' not in part and '+' not in part and 'contact' not in part.lower():
                        if len(part) > 3:
                            address_parts.append(part)
                return ', '.join(address_parts)
        return ""
    
    def extract_skills(self, text):
        """Extract technical skills with better targeting"""
        # Look for skills section
        skills_section = ""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if 'technical skills' in line.lower() or 'tech stack' in line.lower():
                # Get next few lines after skills header
                skills_section = ' '.join(lines[i:i+5])
                break
        
        # If no skills section found, search whole text
        if not skills_section:
            skills_section = text
        
        # Comprehensive technical skills list
        technical_skills = [
            'Flutter', 'Android', 'iOS', 'React', 'Angular', 'Spring', 'Spring Boot',
            'Node.js', 'NodeJs', 'JavaScript', 'Python', 'Java', 'C++', 'C#',
            'MySQL', 'PostgreSQL', 'MongoDB', 'Firebase', 'AWS', 'Docker',
            'Git', 'HTML', 'CSS', 'REST API', 'RESTful', 'Machine Learning',
            'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy',
            'WordPress', 'SQLite', 'Colab', 'Android Studio'
        ]
        
        found_skills = []
        skills_lower = skills_section.lower()
        
        for skill in technical_skills:
            if skill.lower() in skills_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))
    
    def extract_education(self, text):
        """Extract education with precise patterns"""
        education = []
        
        # Look for degree patterns
        degree_patterns = [
            r'Bachelor\s+of\s+[\w\s]+',
            r'Master\s+of\s+[\w\s]+',
            r'PhD\s+in\s+[\w\s]+',
            r'B\.?[A-Z][a-z]*\.?\s+[\w\s]*',
            r'M\.?[A-Z][a-z]*\.?\s+[\w\s]*',
            r'GCE\s+Advanced\s+Level',
            r'Advanced\s+Level'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education.extend(matches)
        
        # Look for university/college names
        institution_patterns = [
            r'University\s+of\s+[\w\s]+',
            r'[\w\s]+\s+University',
            r'[\w\s]+\s+College',
            r'[\w\s]+\s+Institute'
        ]
        
        institutions = []
        for pattern in institution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            institutions.extend(matches)
        
        # Extract GPA if mentioned
        gpa_pattern = r'(?:GPA|cGPA)[:\s]*([0-9]\.[0-9]+)'
        gpa_matches = re.findall(gpa_pattern, text, re.IGNORECASE)
        
        return {
            "degrees": list(set([deg.strip() for deg in education])),
            "institutions": list(set([inst.strip() for inst in institutions])),
            "gpa": gpa_matches
        }
    
    def extract_experience(self, text):
        """Extract work experience with better patterns"""
        # Experience duration patterns
        exp_patterns = [
            r'over\s+(\w+)\s+year[s]?\s+of\s+experience',
            r'(\d+)\+?\s+year[s]?\s+(?:of\s+)?experience',
            r'(\d+)\s+year[s]?\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\s+year[s]?'
        ]
        
        experience_years = []
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experience_years.extend(matches)
        
        # Job titles and companies
        job_patterns = [
            r'Software\s+Engineer\s+at\s+([\w\s\.]+)',
            r'([\w\s]+)\s+at\s+([\w\s\.]+)',
        ]
        
        jobs = []
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            jobs.extend(matches)
        
        return {
            "years_of_experience": list(set(experience_years)),
            "positions": jobs
        }
    
    def extract_projects(self, text):
        """Extract project information"""
        projects = []
        
        # Look for project section
        lines = text.split('\n')
        in_projects = False
        current_project = ""
        
        for line in lines:
            if 'projects' in line.lower() and len(line.strip()) < 20:
                in_projects = True
                continue
            
            if in_projects:
                if 'tech stack' in line.lower():
                    if current_project:
                        projects.append(current_project.strip())
                    current_project = ""
                    projects.append(line.strip())
                elif line.strip() and not line.startswith(' ' * 8):
                    current_project += " " + line.strip()
        
        return projects[:5]  # Limit to 5 projects
    
    def parse_cv(self, file_path):
        """Main parsing function"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(str(file_path))
            else:
                return {"error": f"Unsupported file format: {file_path.suffix}"}
            
            if len(text.strip()) < 50:
                return {"error": "File appears to be empty"}
            
            # Extract all information
            name = self.extract_name(text)
            contact_info = self.extract_contact_info(text)
            skills = self.extract_skills(text)
            education = self.extract_education(text)
            experience = self.extract_experience(text)
            projects = self.extract_projects(text)
            
            result = {
                "personal_info": {
                    "name": name,
                    "contact": contact_info
                },
                "skills": skills,
                "education": education,
                "experience": experience,
                "projects": projects,
                "file_info": {
                    "filename": file_path.name,
                    "text_length": len(text)
                },
                "extraction_metadata": {
                    "extraction_successful": True,
                    "parser_version": "improved_cv_parser_v2.0"
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Failed to parse CV: {str(e)}",
                "extraction_metadata": {
                    "extraction_successful": False,
                    "parser_version": "improved_cv_parser_v2.0"
                }
            }
    
    def parse_cv_to_json(self, file_path, output_path=None):
        result = self.parse_cv(file_path)
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"Results saved to {output_path}")
        
        return json_output

# Usage
def main():
    parser = ImprovedCVParser()
    result_json = parser.parse_cv_to_json("F:/Cogntix/Unblit/AI/SampleCVs/Sample1.pdf", "improved_parsed_cv.json")
    print("Improved CV parsing completed!")
    print(result_json)

if __name__ == "__main__":
    main()
