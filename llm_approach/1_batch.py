# batch_cv_parser.py
import ollama
import json
import fitz  # PyMuPDF
from pathlib import Path
import os
import re
import time
import csv
from typing import List, Dict

class WindowsCVParser:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = model_name
        self.client = ollama.Client()
        print(f"Using model: {model_name}")
        
        # Pre-warm the model for faster processing
        self._warm_model()
    
    def _warm_model(self):
        """Pre-warm model to avoid cold start delays"""
        try:
            self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                options={"num_predict": 1}
            )
            print("Model warmed up!")
        except Exception as e:
            print(f"Warning: Could not warm model: {e}")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def create_parsing_prompt(self, resume_text):
        """Create structured prompt for CV parsing"""
        prompt = f"""Extract key info from this resume as JSON:
{{
    "name": "full name",
    "email": "email", 
    "phone": "phone",
    "experience": [{{"position": "title", "company": "name", "duration": "dates"}}],
    "education": [{{"degree": "degree", "school": "institution", "year": "year"}}],
    "skills": ["skill1", "skill2"],
    "years_experience": "number"
}}

Resume: {resume_text[:3000]}

Return only JSON:"""
        return prompt

    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response regardless of formatting"""
        text = response_text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r'^```.*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Remove "json" prefix if present
        text = re.sub(r'^json\s*', '', text, flags=re.IGNORECASE)
        
        # Find JSON content using regex
        json_pattern = r'(\{.*\}|\[.*\])'
        match = re.search(json_pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return text.strip()
    
    def parse_cv(self, pdf_path):
        """Main parsing function"""
        try:
            start_time = time.time()
            
            # Extract text
            resume_text = self.extract_text_from_pdf(pdf_path)
            if "Error reading PDF" in resume_text:
                return {"error": resume_text}
            
            # Create prompt
            prompt = self.create_parsing_prompt(resume_text)
            
            # Get LLM response with optimized parameters
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "num_predict": 1500,
                    "num_thread": 8,
                    "repeat_penalty": 1.0,
                    "top_k": 10,
                }
            )
            
            # Extract and clean response
            llm_output = response['message']['content']
            clean_json = self.extract_json_from_response(llm_output)
        
            # Parse JSON
            parsed_data = json.loads(clean_json)
            
            # Add metadata
            parsed_data["extraction_metadata"] = {
                "method": "LLaMA_Windows",
                "model": self.model_name,
                "success": True,
                "file_processed": os.path.basename(pdf_path),
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": llm_output if 'llm_output' in locals() else "No response",
                "cleaned_json": clean_json if 'clean_json' in locals() else "N/A"
            }
        except Exception as e:
            return {"error": f"Processing failed: {str(e)}"}

class BatchCVProcessor:
    def __init__(self, model_name="llama3.2:1b"):
        self.parser = WindowsCVParser(model_name)
        
    def format_experience_data(self, experience_list):
        """Format experience data for CSV"""
        if not experience_list or not isinstance(experience_list, list):
            return ""
        
        formatted_experiences = []
        for exp in experience_list:
            if isinstance(exp, dict):
                position = exp.get('position', '')
                company = exp.get('company', '')
                duration = exp.get('duration', '')
                formatted_experiences.append(f"{position} at {company} ({duration})")
        
        return " | ".join(formatted_experiences)
    
    def format_education_data(self, education_list):
        """Format education data for CSV"""
        if not education_list or not isinstance(education_list, list):
            return ""
        
        formatted_education = []
        for edu in education_list:
            if isinstance(edu, dict):
                degree = edu.get('degree', '')
                school = edu.get('school', '')
                year = edu.get('year', '')
                formatted_education.append(f"{degree} from {school} ({year})")
        
        return " | ".join(formatted_education)
    
    def format_skills_data(self, skills_list):
        """Format skills data for CSV"""
        if not skills_list:
            return ""
        
        if isinstance(skills_list, list):
            return ", ".join(skills_list)
        else:
            return str(skills_list)
    
    def process_folder(self, input_folder: str, output_csv_path: str):
        """Process all PDFs in a folder and save to CSV"""
        
        # Check if input folder exists
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist!")
            return
        
        # Find all PDF files
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in '{input_folder}'")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # CSV columns
        csv_columns = [
            "File_Name",
            "Name", 
            "Email", 
            "Phone",
            "Years_Experience",
            "Experience_Details",
            "Education_Details", 
            "Skills",
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
            result = self.parser.parse_cv(str(pdf_path))
            processing_time = time.time() - start_time
            
            # Prepare row data
            row_data = {
                "File_Name": pdf_path.name,
                "Processing_Time_Seconds": round(processing_time, 2)
            }
            
            if "error" in result:
                # Handle errors
                row_data.update({
                    "Name": "",
                    "Email": "",
                    "Phone": "",
                    "Years_Experience": "",
                    "Experience_Details": "",
                    "Education_Details": "",
                    "Skills": "",
                    "Status": f"ERROR: {result['error'][:100]}"
                })
                failed_parses += 1
                print(f"Error: {result['error'][:100]}")
            else:
                # Handle successful parsing
                row_data.update({
                    "Name": result.get('name', ''),
                    "Email": result.get('email', ''),
                    "Phone": result.get('phone', ''),
                    "Years_Experience": result.get('years_experience', ''),
                    "Experience_Details": self.format_experience_data(result.get('experience', [])),
                    "Education_Details": self.format_education_data(result.get('education', [])),
                    "Skills": self.format_skills_data(result.get('skills', [])),
                    "Status": "SUCCESS"
                })
                successful_parses += 1
                print(f"Success: {result.get('name', 'Unknown')} - {processing_time:.2f}s")
            
            results.append(row_data)
        
        # Save to CSV
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n=== BATCH PROCESSING COMPLETE ===")
            print(f"Total files processed: {len(pdf_files)}")
            print(f"Successful parses: {successful_parses}")
            print(f"Failed parses: {failed_parses}")
            print(f"CSV saved to: {output_csv_path}")
            print(f"Success rate: {(successful_parses/len(pdf_files)*100):.1f}%")
            
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")
    
    def process_single_folder_quick_summary(self, input_folder: str):
        """Quick summary of files in folder without full processing"""
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files:")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            file_size = pdf_path.stat().st_size / 1024  # KB
            print(f"  {i}. {pdf_path.name} ({file_size:.1f} KB)")

def main():
    print("=== Batch CV Parser ===")
    
    # Configuration
    input_folder = "F:/Cogntix/Unblit/AI/SampleCVs/"
    output_csv = "F:/Cogntix/Unblit/AI/SampleCVs/parsed_cvs_results.csv"
    model_name = "llama3.2:1b"  # Fast model for batch processing
    
    # Initialize processor
    processor = BatchCVProcessor(model_name)
    
    # Show quick summary first
    print("\nScanning folder...")
    processor.process_single_folder_quick_summary(input_folder)
    
    # Ask for confirmation
    response = input(f"\nProcess all PDFs and save to CSV? (y/n): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return
    
    # Process all files
    print("\nStarting batch processing...")
    start_time = time.time()
    
    processor.process_folder(input_folder, output_csv)
    
    total_time = time.time() - start_time
    print(f"\nTotal batch processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
