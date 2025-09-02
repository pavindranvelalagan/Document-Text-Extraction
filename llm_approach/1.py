# windows_cv_parser.py
import ollama
import json
import fitz  # PyMuPDF
from pathlib import Path
import os
import re
import time

class WindowsCVParser:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.client = ollama.Client()
        print(f"Using model: {model_name}")
    
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
        # Remove common prefixes
        text = response_text.strip()
        
        # Remove markdown code blocks - FIXED REGEX PATTERNS
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
            print(f"Processing: {pdf_path}")
            
            # Extract text
            resume_text = self.extract_text_from_pdf(pdf_path)
            if "Error reading PDF" in resume_text:
                return {"error": resume_text}
            
            print(f"Extracted {len(resume_text)} characters from PDF")
            
            # Create prompt
            prompt = self.create_parsing_prompt(resume_text)
            
            print("Sending to LLaMA for processing...")
            
            # Get LLM response with optimized parameters
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.0,        # Faster than 0.1
                    "top_p": 0.9,             # Focus on top tokens
                    "num_predict": 1500,      # Reduced from 2000
                    "num_thread": 8,          # Use more CPU threads
                    "repeat_penalty": 1.0,    # Disable repeat penalty
                    "top_k": 10,             # Limit vocabulary consideration
                }
            )
            
            # Extract and clean response using robust method
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
    
    def save_results(self, result, output_path):
        """Save results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")

def main():
    # Initialize parser with different models for speed testing
    
    # test 1: took 26.08 seconds
    # parser = WindowsCVParser("llama3.2") 
    
    # test 2: took 8.02 seconds - FASTEST SO FAR
    # test 2: took 4.77 seconds - After optimization 
    parser = WindowsCVParser("llama3.2:1b")  
    
    # test 3: took 30 seconds and returned empty json with error 
    # parser = WindowsCVParser("gemma2:2b")
    
    # test 4: took 18.64 seconds
    # parser = WindowsCVParser("qwen2.5:3b")
    
    # Process CV
    cv_file = "F:/Cogntix/Unblit/AI/SampleCVs/Sample1.pdf" 
    output_file = "llama_parsed_cv.json"
    
    if not os.path.exists(cv_file):
        print(f"Error: CV file '{cv_file}' not found!")
        print("Please update the cv_file variable with the correct path to your CV.")
        return
    
    print("=== LLaMA CV Parser for Windows ===")
    start_time = time.time()
    result = parser.parse_cv(cv_file)
    total_time = time.time() - start_time
    
    # Save results
    parser.save_results(result, output_file)
    
    # Display summary
    if "error" not in result:
        print("\n=== PARSING SUMMARY ===")
        print(f"Name: {result.get('name', 'Not found')}")
        print(f"Email: {result.get('email', 'Not found')}")
        print(f"Phone: {result.get('phone', 'Not found')}")
        print(f"Skills: {len(result.get('skills', []))}")
        print(f"Experience: {len(result.get('experience', []))} positions")
        print(f"Education: {len(result.get('education', []))} entries")
        print(f"Years Experience: {result.get('years_experience', 'Not found')}")
        print(f"Total Processing Time: {total_time:.2f} seconds")
    else:
        print(f"Error: {result['error']}")
        if 'raw_response' in result:
            print(f"Raw Response: {result['raw_response'][:200]}...")

if __name__ == "__main__":
    main()
