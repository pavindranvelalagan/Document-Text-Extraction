# batch_multicolumn_parser.py
import fitz
import re
import json
import spacy
from pathlib import Path
import csv
import os
import time
from typing import List, Dict

class MultiColumnCVParser:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_with_layout_preservation(self, pdf_path):
        """Extract text preserving multi-column layout"""
        doc = fitz.open(pdf_path)
        page = doc[0]
        blocks = page.get_text("blocks")
        page_width = page.rect.width
        
        # Detect columns
        column_boundaries = self._detect_column_boundaries(blocks, page_width)
        
        # Separate blocks by columns
        columns = self._separate_into_columns(blocks, column_boundaries)
        
        # Parse each column separately
        parsed_columns = []
        for i, column_blocks in enumerate(columns):
            column_text = self._combine_column_text(column_blocks)
            parsed_data = self._parse_column_content(column_text, f"Column_{i+1}")
            parsed_columns.append(parsed_data)
        
        # Intelligently merge column data
        merged_result = self._merge_column_data(parsed_columns)
        
        doc.close()
        return merged_result
    
    def _detect_column_boundaries(self, blocks, page_width):
        """Smart column boundary detection"""
        x_positions = []
        for block in blocks:
            x0, y0, x1, y1, text, _, _ = block
            if len(text.strip()) > 10:  # Only consider substantial text blocks
                x_positions.append(x0)
        
        if len(x_positions) < 2:
            return [page_width / 2]
        
        x_positions.sort()
        
        # Find significant gaps (potential column separators)
        gaps = []
        for i in range(len(x_positions) - 1):
            gap = x_positions[i + 1] - x_positions[i]
            if gap > 40:  # Minimum gap for column separation
                gaps.append((gap, (x_positions[i] + x_positions[i + 1]) / 2))
        
        if gaps:
            # Use the largest gap as column boundary
            largest_gap = max(gaps, key=lambda x: x[0])
            return [largest_gap[1]]
        
        return [page_width / 2]  # Default fallback
    
    def _separate_into_columns(self, blocks, boundaries):
        """Separate blocks into columns and sort by position"""
        num_columns = len(boundaries) + 1
        columns = [[] for _ in range(num_columns)]
        
        for block in blocks:
            x0, y0, x1, y1, text, _, _ = block
            
            if len(text.strip()) < 3:
                continue
            
            # Determine column based on x-position
            block_center = (x0 + x1) / 2
            column_idx = 0
            
            for i, boundary in enumerate(boundaries):
                if block_center > boundary:
                    column_idx = i + 1
            
            columns[column_idx].append({
                'text': text,
                'y_pos': y0,
                'coords': (x0, y0, x1, y1)
            })
        
        # Sort each column by vertical position
        for column in columns:
            column.sort(key=lambda x: x['y_pos'])
        
        return columns
    
    def _combine_column_text(self, column_blocks):
        """Combine text blocks in a column while preserving structure"""
        return '\n'.join([block['text'] for block in column_blocks])
    
    def _parse_column_content(self, text, column_name):
        """Parse content from a single column"""
        result = {
            'column_name': column_name,
            'raw_text': text,
            'sections': self._identify_sections(text),
            'contact_info': self._extract_contact_info(text),
            'skills': self._extract_skills(text),
            'education': self._extract_education_smart(text),
            'experience': self._extract_experience_smart(text),
            'projects': self._extract_projects(text)
        }
        return result
    
    def _identify_sections(self, text):
        """Identify different sections in the text"""
        sections = {}
        lines = text.split('\n')
        
        current_section = "unknown"
        section_content = []
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Check if this line is a section header
            if self._is_section_header(line_clean):
                # Save previous section
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                
                # Start new section
                current_section = self._normalize_section_name(line_clean)
                section_content = []
            else:
                section_content.append(line_clean)
        
        # Save last section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    def _is_section_header(self, line):
        """Check if a line is likely a section header"""
        header_keywords = [
            'summary', 'experience', 'education', 'skills', 'projects',
            'work experience', 'employment', 'qualifications', 'achievements',
            'contact', 'profile', 'objective', 'career', 'background'
        ]
        
        line_lower = line.lower()
        
        # Check if line is short and contains header keywords
        if len(line) < 50 and any(keyword in line_lower for keyword in header_keywords):
            return True
        
        # Check if line is all caps (common for headers)
        if line.isupper() and len(line.split()) <= 3:
            return True
        
        return False
    
    def _normalize_section_name(self, header):
        """Normalize section header names"""
        header_lower = header.lower()
        
        if any(word in header_lower for word in ['experience', 'work', 'employment']):
            return 'experience'
        elif any(word in header_lower for word in ['education', 'qualification']):
            return 'education'
        elif any(word in header_lower for word in ['skill', 'technical']):
            return 'skills'
        elif any(word in header_lower for word in ['project', 'portfolio']):
            return 'projects'
        elif any(word in header_lower for word in ['summary', 'profile', 'objective']):
            return 'summary'
        else:
            return header_lower.replace(' ', '_')
    
    def _extract_contact_info(self, text):
        """Extract contact information with improved patterns"""
        contact = {
            'emails': [],
            'phones': [],
            'linkedin': [],
            'github': [],
            'address': ''
        }
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        contact['emails'] = list(set(re.findall(email_pattern, text)))
        
        # Phone extraction - multiple patterns
        phone_patterns = [
            r'\+\d{2}\(\d\)\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}',  # +94(0) 76 97 87 765
            r'\+\d{2}\s?\d{2}\s?\d{7,10}',  # +94 11 2650301
            r'\(\d{3}\)\s?\d{3}[-\s]?\d{4}',  # (123) 456-7890
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        contact['phones'] = list(set(phones))
        
        # LinkedIn
        linkedin_patterns = [
            r'linkedin\.com/in/[\w-]+',
            r'LinkedIn profile'
        ]
        
        for pattern in linkedin_patterns:
            contact['linkedin'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        return contact
    
    def _extract_skills(self, text):
        """Extract technical skills with section awareness"""
        # Look for skills section first
        skills_section = ""
        lines = text.split('\n')
        
        in_skills_section = False
        for i, line in enumerate(lines):
            if 'technical skills' in line.lower() or 'key skills' in line.lower():
                in_skills_section = True
                continue
            elif in_skills_section and self._is_section_header(line):
                break
            elif in_skills_section:
                skills_section += line + " "
        
        if not skills_section:
            skills_section = text  # Fallback to full text
        
        # Comprehensive skills list
        technical_skills = [
            'Python', 'Java', 'JavaScript', 'React', 'Angular', 'Vue.js',
            'Node.js', 'Express', 'Django', 'Flask', 'Spring Boot',
            'HTML', 'CSS', 'TypeScript', 'PHP', 'Ruby', 'C++', 'C#',
            'Flutter', 'React Native', 'Android', 'iOS', 'Swift', 'Kotlin',
            'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite',
            'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes',
            'Git', 'Jenkins', 'CI/CD', 'Linux', 'Unix',
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch',
            'Pandas', 'NumPy', 'Scikit-learn', 'OpenCV',
            'REST API', 'GraphQL', 'Microservices', 'Agile', 'Scrum'
        ]
        
        found_skills = []
        skills_lower = skills_section.lower()
        
        for skill in technical_skills:
            if skill.lower() in skills_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))
    
    def _extract_education_smart(self, text):
        """Smart education extraction"""
        education = {
            'degrees': [],
            'institutions': [],
            'gpa': [],
            'years': []
        }
        
        # Degree patterns
        degree_patterns = [
            r'Bachelor\s+of\s+[\w\s]+',
            r'Master\s+of\s+[\w\s]+',
            r'PhD\s+in\s+[\w\s]+',
            r'B\.?[A-Z][a-z]*\.?\s+[\w\s]*',
            r'M\.?[A-Z][a-z]*\.?\s+[\w\s]*',
            r'GCE\s+Advanced\s+Level'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education['degrees'].extend([m.strip() for m in matches])
        
        # Institution patterns
        institution_patterns = [
            r'University\s+of\s+[\w\s]+',
            r'[\w\s]+\s+University',
            r'[\w\s]+\s+College'
        ]
        
        for pattern in institution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education['institutions'].extend([m.strip() for m in matches])
        
        # GPA extraction
        gpa_pattern = r'(?:GPA|cGPA)[:\s]*([0-9]\.[0-9]+)'
        education['gpa'] = re.findall(gpa_pattern, text, re.IGNORECASE)
        
        return education
    
    def _extract_experience_smart(self, text):
        """Smart work experience extraction"""
        experience = {
            'total_years': [],
            'positions': [],
            'companies': [],
            'durations': []
        }
        
        # Years of experience
        exp_patterns = [
            r'over\s+(\w+)\s+year[s]?\s+of\s+experience',
            r'(\d+)\+?\s+year[s]?\s+(?:of\s+)?experience'
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experience['total_years'].extend(matches)
        
        # Job positions and companies
        job_patterns = [
            r'([\w\s]+(?:Engineer|Developer|Manager|Analyst|Consultant))\s+at\s+([\w\s\.]+)',
            r'(Internship)\s+at\s+([\w\s\.]+)'
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for position, company in matches:
                experience['positions'].append(position.strip())
                experience['companies'].append(company.strip())
        
        return experience
    
    def _extract_projects(self, text):
        """Extract project information"""
        projects = []
        
        # Look for project section
        lines = text.split('\n')
        in_projects = False
        current_project = ""
        
        for line in lines:
            line = line.strip()
            if 'project' in line.lower() and len(line) < 50:
                in_projects = True
                if current_project:
                    projects.append(current_project.strip())
                    current_project = ""
            elif in_projects and line:
                if 'tech stack' in line.lower():
                    if current_project:
                        projects.append(current_project.strip())
                    projects.append(line)
                    current_project = ""
                else:
                    current_project += " " + line
        
        if current_project:
            projects.append(current_project.strip())
        
        return projects[:5]  # Limit to 5 projects
    
    def _merge_column_data(self, parsed_columns):
        """Intelligently merge data from multiple columns"""
        merged = {
            'personal_info': {
                'contact': {},
                'name': ''
            },
            'skills': [],
            'education': {
                'degrees': [],
                'institutions': [],
                'gpa': []
            },
            'experience': {
                'total_years': [],
                'positions': [],
                'companies': []
            },
            'projects': [],
            'column_analysis': []
        }
        
        all_contacts = {}
        all_skills = []
        all_education = {'degrees': [], 'institutions': [], 'gpa': []}
        all_experience = {'total_years': [], 'positions': [], 'companies': []}
        all_projects = []
        
        for column in parsed_columns:
            # Merge contact info
            for key, value in column['contact_info'].items():
                if value:
                    if key not in all_contacts:
                        all_contacts[key] = []
                    if isinstance(value, list):
                        all_contacts[key].extend(value)
                    else:
                        all_contacts[key].append(value)
            
            # Merge skills
            all_skills.extend(column['skills'])
            
            # Merge education
            for key in all_education:
                all_education[key].extend(column['education'][key])
            
            # Merge experience
            for key in all_experience:
                all_experience[key].extend(column['experience'][key])
            
            # Merge projects
            all_projects.extend(column['projects'])
            
            # Keep column analysis for debugging
            merged['column_analysis'].append({
                'column': column['column_name'],
                'sections_found': list(column['sections'].keys()),
                'text_length': len(column['raw_text'])
            })
        
        # Deduplicate and clean
        merged['personal_info']['contact'] = {k: list(set(v)) for k, v in all_contacts.items()}
        merged['skills'] = list(set(all_skills))
        merged['education'] = {k: list(set(v)) for k, v in all_education.items()}
        merged['experience'] = {k: list(set(v)) for k, v in all_experience.items()}
        merged['projects'] = list(set(all_projects))
        
        # Extract name from contact or first substantial text
        if merged['column_analysis']:
            first_column = parsed_columns[0]
            lines = first_column['raw_text'].split('\n')[:3]
            for line in lines:
                line = line.strip()
                if len(line) > 5 and not '@' in line and not '+' in line:
                    if re.match(r'^[A-Za-z\s]+$', line):
                        merged['personal_info']['name'] = line
                        break
        
        return merged

class BatchMultiColumnProcessor:
    def __init__(self):
        self.parser = MultiColumnCVParser()
        
    def format_contact_data(self, contact_dict):
        """Format contact data for CSV"""
        emails = ', '.join(contact_dict.get('emails', []))
        phones = ', '.join(contact_dict.get('phones', []))
        linkedin = ', '.join(contact_dict.get('linkedin', []))
        github = ', '.join(contact_dict.get('github', []))
        
        return emails, phones, linkedin, github
    
    def format_experience_data(self, experience_dict):
        """Format experience data for CSV"""
        positions = experience_dict.get('positions', [])
        companies = experience_dict.get('companies', [])
        
        # Pair positions with companies
        experiences = []
        for i in range(max(len(positions), len(companies))):
            pos = positions[i] if i < len(positions) else "Unknown Position"
            comp = companies[i] if i < len(companies) else "Unknown Company"
            experiences.append(f"{pos} at {comp}")
        
        return ' | '.join(experiences)
    
    def format_education_data(self, education_dict):
        """Format education data for CSV"""
        degrees = education_dict.get('degrees', [])
        institutions = education_dict.get('institutions', [])
        gpas = education_dict.get('gpa', [])
        
        educations = []
        for i in range(max(len(degrees), len(institutions))):
            deg = degrees[i] if i < len(degrees) else "Unknown Degree"
            inst = institutions[i] if i < len(institutions) else "Unknown Institution"
            educations.append(f"{deg} from {inst}")
        
        gpa_str = f" (GPA: {', '.join(gpas)})" if gpas else ""
        return ' | '.join(educations) + gpa_str
    
    def process_folder(self, input_folder: str, output_csv_path: str):
        """Process all PDFs in a folder and save to CSV"""
        
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist!")
            return
        
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in '{input_folder}'")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process with Multi-Column Parser")
        
        # CSV columns
        csv_columns = [
            "File_Name",
            "Name", 
            "Emails", 
            "Phones",
            "LinkedIn",
            "GitHub",
            "Skills",
            "Experience_Details",
            "Education_Details",
            "Projects",
            "Column_Count",
            "Processing_Time_Seconds",
            "Status"
        ]
        
        # Process each PDF
        results = []
        successful_parses = 0
        failed_parses = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_path.name}")
            
            start_time = time.time()
            
            try:
                result = self.parser.extract_with_layout_preservation(str(pdf_path))
                processing_time = time.time() - start_time
                
                # Extract data
                name = result['personal_info']['name']
                contact = result['personal_info']['contact']
                emails, phones, linkedin, github = self.format_contact_data(contact)
                
                skills = ', '.join(result['skills'])
                experience = self.format_experience_data(result['experience'])
                education = self.format_education_data(result['education'])
                projects = ' | '.join(result['projects'][:3])  # Limit to 3 projects
                
                column_count = len(result['column_analysis'])
                
                row_data = {
                    "File_Name": pdf_path.name,
                    "Name": name,
                    "Emails": emails,
                    "Phones": phones,
                    "LinkedIn": linkedin,
                    "GitHub": github,
                    "Skills": skills,
                    "Experience_Details": experience,
                    "Education_Details": education,
                    "Projects": projects,
                    "Column_Count": column_count,
                    "Processing_Time_Seconds": round(processing_time, 2),
                    "Status": "SUCCESS"
                }
                
                successful_parses += 1
                print(f"Success: {name} - {processing_time:.2f}s - {column_count} columns")
                
            except Exception as e:
                processing_time = time.time() - start_time
                row_data = {
                    "File_Name": pdf_path.name,
                    "Name": "",
                    "Emails": "",
                    "Phones": "",
                    "LinkedIn": "",
                    "GitHub": "",
                    "Skills": "",
                    "Experience_Details": "",
                    "Education_Details": "",
                    "Projects": "",
                    "Column_Count": 0,
                    "Processing_Time_Seconds": round(processing_time, 2),
                    "Status": f"ERROR: {str(e)[:100]}"
                }
                
                failed_parses += 1
                print(f"Error: {str(e)[:100]}")
            
            results.append(row_data)
        
        # Save to CSV
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n=== MULTI-COLUMN BATCH PROCESSING COMPLETE ===")
            print(f"Total files processed: {len(pdf_files)}")
            print(f"Successful parses: {successful_parses}")
            print(f"Failed parses: {failed_parses}")
            print(f"CSV saved to: {output_csv_path}")
            print(f"Success rate: {(successful_parses/len(pdf_files)*100):.1f}%")
            
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")

def main():
    print("=== Multi-Column CV Batch Parser ===")
    
    # Configuration
    input_folder = "F:/Cogntix/Unblit/AI/SampleCVs/"
    output_csv = "F:/Cogntix/Unblit/AI/SampleCVs/multicolumn_parsed_results.csv"
    
    # Initialize processor
    processor = BatchMultiColumnProcessor()
    
    # Ask for confirmation
    response = input(f"\nProcess all PDFs with Multi-Column Parser? (y/n): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return
    
    # Process all files
    print("\nStarting multi-column batch processing...")
    start_time = time.time()
    
    processor.process_folder(input_folder, output_csv)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
