# llm_cv_parser.py
import ollama
import json
import fitz  # For PDF text extraction
from pathlib import Path

class LLaMACVParser:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Verify model is available
        try:
            self.client.chat(model=model_name, messages=[{"role": "user", "content": "test"}])
        except:
            print(f"Model {model_name} not found. Installing...")
            ollama.pull(model_name)
    
    def extract_text_from_pdf(self, pdf_path):
        """Simple text extraction - no coordinate complexity needed"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def create_parsing_prompt(self, resume_text, job_title="Software Engineer"):
        """Create a structured prompt for CV parsing"""
        prompt = f"""
You are an expert HR assistant. Extract information from this resume and return ONLY a valid JSON object with the following structure:

{{
  "personal_info": {{
    "name": "Full name",
    "email": "email@example.com",
    "phone": "phone number",
    "address": "address if available",
    "linkedin": "linkedin profile if available",
    "github": "github profile if available"
  }},
  "professional_summary": "Brief summary of candidate's profile",
  "experience": [
    {{
      "position": "Job title",
      "company": "Company name",
      "duration": "Start - End dates",
      "description": "Key responsibilities and achievements"
    }}
  ],
  "education": [
    {{
      "degree": "Degree name",
      "institution": "University/College name",
      "year": "Graduation year",
      "gpa": "GPA if mentioned"
    }}
  ],
  "skills": {{
    "technical_skills": ["skill1", "skill2", "skill3"],
    "soft_skills": ["skill1", "skill2"],
    "programming_languages": ["language1", "language2"],
    "frameworks_tools": ["tool1", "tool2"]
  }},
  "projects": [
    {{
      "name": "Project name",
      "description": "Project description",
      "technologies": ["tech1", "tech2"],
      "role": "Your role in project"
    }}
  ],
  "certifications": ["cert1", "cert2"],
  "years_of_experience": "Total years of experience as number",
  "job_relevance_score": "Score from 1-10 for {job_title} position"
}}

Resume text:
{resume_text}

Return ONLY the JSON object, no additional text or explanation.
"""
        return prompt
    
    def parse_resume(self, pdf_path, job_title="Software Engineer"):
        """Main parsing function using LLaMA"""
        try:
            # Extract text (simple, no coordinate complexity)
            resume_text = self.extract_text_from_pdf(pdf_path)
            
            if not resume_text.strip():
                return {"error": "Could not extract text from PDF"}
            
            # Create prompt
            prompt = self.create_parsing_prompt(resume_text, job_title)
            
            # Get LLM response
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,  # Low temperature for consistent output
                    "num_predict": 2000   # Limit response length
                }
            )
            
            # Extract JSON from response
            llm_output = response['message']['content'].strip()
            
            # Clean the response (remove any markdown formatting)
            if llm_output.startswith("```
                llm_output = llm_output[7:-3]
            elif llm_output.startswith("```"):
                llm_output = llm_output[3:-3]
            
            # Parse JSON
            parsed_data = json.loads(llm_output)
            
            # Add metadata
            parsed_data["extraction_metadata"] = {
                "method": "LLaMA_local",
                "model": self.model_name,
                "success": True,
                "text_length": len(resume_text)
            }
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
                "raw_response": llm_output if 'llm_output' in locals() else "No response"
            }
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}
    
    def parse_to_json_file(self, pdf_path, output_path, job_title="Software Engineer"):
        """Parse and save to JSON file"""
        result = self.parse_resume(pdf_path, job_title)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
        return result

# Usage example
def main():
    parser = LLaMACVParser("llama3.2")
    
    result = parser.parse_to_json_file(
        "Sample1.pdf", 
        "llama_parsed_cv.json", 
        "Software Engineer"
    )
    
    print("Parsing completed!")
    print(json.dumps(result, indent=2)[:500] + "...")

if __name__ == "__main__":
    main()
