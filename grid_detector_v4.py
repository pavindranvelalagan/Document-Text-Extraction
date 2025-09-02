# enhanced_histogram_cv_parser.py
import pdfplumber
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import csv
import re
import time
from pathlib import Path
from typing import List, Dict
import json

class HistogramCVParser:
    def __init__(self):
        self.contact_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'[\+]?[\d\s\-\(\)]{10,}',
            'linkedin': r'linkedin\.com/[\w\-/]+',
            'github': r'github\.com/[\w\-/]+'
        }
        
        self.skills_keywords = [
            'Python', 'Java', 'JavaScript', 'React', 'Angular', 'Vue',
            'Node.js', 'Django', 'Flask', 'Spring', 'HTML', 'CSS',
            'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'AWS', 'Azure',
            'Docker', 'Kubernetes', 'Git', 'Machine Learning', 'AI',
            'Flutter', 'Android', 'iOS', 'Swift', 'Kotlin', 'C++', 'C#'
        ]
    
    def clip_bbox(self, bbox, max_width, max_height):
        """Ensure bounding box stays within image boundaries"""
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(x0, max_width))
        x1 = max(0, min(x1, max_width))
        y0 = max(0, min(y0, max_height))
        y1 = max(0, min(y1, max_height))
        
        if x0 >= x1: x1 = x0 + 1
        if y0 >= y1: y1 = y0 + 1
        
        return (x0, y0, x1, y1)
    
    def compute_histograms(self, cv_img, smooth_kernel=5):
        """Compute vertical and horizontal histograms for layout detection"""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Different smoothing for different directions
        v_smoothed = cv2.blur(binary, (3, 1))  # Preserve column gaps
        h_smoothed = cv2.blur(binary, (1, smooth_kernel))  # Smooth rows
        
        vertical_hist = np.sum(v_smoothed, axis=0)
        horizontal_hist = np.sum(h_smoothed, axis=1)
        
        return vertical_hist, horizontal_hist, binary
    
    def find_separators(self, histogram, min_gap_ratio=0.03, threshold_ratio=0.08, margin_ratio=0.05):
        """Find gaps in histogram that indicate column/row separators"""
        total_length = len(histogram)
        max_density = np.max(histogram)
        
        if max_density == 0:
            return []
        
        threshold = max_density * threshold_ratio
        min_gap_width = int(total_length * min_gap_ratio)
        margin_ignore = int(total_length * margin_ratio)
        
        # Find low-density areas
        below_threshold = np.where(histogram < threshold)[0]
        
        if len(below_threshold) == 0:
            return []
        
        # Group consecutive positions into gaps
        gaps = []
        start = below_threshold[0]
        prev = below_threshold[0]
        
        for pos in below_threshold[1:]:
            if pos == prev + 1:
                prev = pos
            else:
                # Check if gap is significant
                width = prev - start + 1
                if (width >= min_gap_width and 
                    start > margin_ignore and 
                    prev < (total_length - margin_ignore)):
                    
                    center = (start + prev) // 2
                    gaps.append({
                        'start': int(start),
                        'end': int(prev),
                        'center': int(center),
                        'width': int(width)
                    })
                
                start = pos
                prev = pos
        
        # Handle last gap
        width = prev - start + 1
        if (width >= min_gap_width and 
            start > margin_ignore and 
            prev < (total_length - margin_ignore)):
            
            center = (start + prev) // 2
            gaps.append({
                'start': int(start),
                'end': int(prev),
                'center': int(center),
                'width': int(width)
            })
        
        return gaps
    
    def create_regions(self, v_separators, h_separators, img_width, img_height, page_width, page_height):
        """Create text regions based on detected separators"""
        # Create boundaries
        x_boundaries = [0]
        for sep in v_separators:
            x_boundaries.append(min(sep['center'], img_width - 1))
        x_boundaries.append(img_width)
        x_boundaries = sorted(list(set(x_boundaries)))
        
        y_boundaries = [0]
        for sep in h_separators:
            y_boundaries.append(min(sep['center'], img_height - 1))
        y_boundaries.append(img_height)
        y_boundaries = sorted(list(set(y_boundaries)))
        
        # Calculate scaling factors
        x_scale = page_width / img_width
        y_scale = page_height / img_height
        
        regions = []
        region_id = 0
        
        for i in range(len(y_boundaries) - 1):
            for j in range(len(x_boundaries) - 1):
                x1, x2 = x_boundaries[j], x_boundaries[j + 1]
                y1, y2 = y_boundaries[i], y_boundaries[i + 1]
                
                # Skip tiny regions
                if (x2 - x1) < 50 or (y2 - y1) < 30:
                    continue
                
                # Clip to image boundaries
                x1, y1, x2, y2 = self.clip_bbox((x1, y1, x2, y2), img_width, img_height)
                
                # Convert to PDF coordinates
                pdf_x1 = x1 * x_scale
                pdf_y1 = y1 * y_scale
                pdf_x2 = x2 * x_scale
                pdf_y2 = y2 * y_scale
                
                # Final clipping
                pdf_bbox = self.clip_bbox((pdf_x1, pdf_y1, pdf_x2, pdf_y2), page_width, page_height)
                
                if pdf_bbox[2] <= pdf_bbox[0] or pdf_bbox[3] <= pdf_bbox[1]:
                    continue
                
                regions.append({
                    'id': region_id,
                    'bbox_pdf': pdf_bbox,
                    'bbox_img': (x1, y1, x2, y2),
                    'row': i,
                    'col': j,
                    'width': x2 - x1,
                    'height': y2 - y1
                })
                region_id += 1
        
        return regions
    
    def extract_region_texts(self, pdf_path, regions, page_num=0):
        """Extract text from each region"""
        region_texts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return []
                
                page = pdf.pages[page_num]
                
                for region in regions:
                    try:
                        pdf_bbox = region['bbox_pdf']
                        
                        # Safety check
                        if (pdf_bbox[0] >= pdf_bbox[2] or pdf_bbox[1] >= pdf_bbox[3] or
                            pdf_bbox[2] > page.width or pdf_bbox[3] > page.height):
                            continue
                        
                        cropped_page = page.crop(pdf_bbox)
                        text = cropped_page.extract_text() or ""
                        text = text.strip()
                        
                        region_texts.append({
                            'region_id': region['id'],
                            'text': text,
                            'row': region['row'],
                            'col': region['col'],
                            'char_count': len(text)
                        })
                        
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        
        return region_texts
    
    def extract_contact_info(self, all_text):
        """Extract contact information from combined text"""
        contact = {}
        
        for contact_type, pattern in self.contact_patterns.items():
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            contact[contact_type] = list(set(matches))
        
        return contact
    
    def extract_skills(self, all_text):
        """Extract skills from text"""
        found_skills = []
        text_lower = all_text.lower()
        
        for skill in self.skills_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))
    
    def extract_name(self, all_text):
        """Extract name from first few lines"""
        lines = all_text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if (len(line) > 5 and len(line) < 50 and 
                not '@' in line and not '+' in line and
                re.match(r'^[A-Za-z\s]+$', line)):
                return line
        return ""
    
    def process_single_cv(self, pdf_path):
        """Process a single CV using histogram-based approach"""
        try:
            print(f"Processing: {os.path.basename(pdf_path)}")
            start_time = time.time()
            
            # Open PDF and get page info
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    return {"error": "No pages found"}
                
                page = pdf.pages[0]
                page_width, page_height = page.width, page.height
                
                # Convert to image for histogram analysis
                img = page.to_image(resolution=200)
                cv_img = cv2.cvtColor(np.array(img.original), cv2.COLOR_RGB2BGR)
                
                img_height, img_width = cv_img.shape[:2]
            
            # Compute histograms
            v_hist, h_hist, binary = self.compute_histograms(cv_img)
            
            # Find separators with optimized parameters
            v_separators = self.find_separators(
                v_hist, 
                min_gap_ratio=0.025,
                threshold_ratio=0.025,
                margin_ratio=0.03
            )
            
            h_separators = self.find_separators(
                h_hist,
                min_gap_ratio=0.05,
                threshold_ratio=0.20,
                margin_ratio=0.05
            )
            
            # Create regions
            regions = self.create_regions(
                v_separators, h_separators, 
                img_width, img_height, 
                page_width, page_height
            )
            
            # Extract text from regions
            region_texts = self.extract_region_texts(pdf_path, regions)
            
            # Combine all text for analysis
            all_text = '\n'.join([rt['text'] for rt in region_texts])
            
            # Extract structured information
            name = self.extract_name(all_text)
            contact_info = self.extract_contact_info(all_text)
            skills = self.extract_skills(all_text)
            
            processing_time = time.time() - start_time
            
            result = {
                'filename': os.path.basename(pdf_path),
                'name': name,
                'emails': ', '.join(contact_info.get('email', [])),
                'phones': ', '.join(contact_info.get('phone', [])),
                'linkedin': ', '.join(contact_info.get('linkedin', [])),
                'github': ', '.join(contact_info.get('github', [])),
                'skills': ', '.join(skills),
                'total_text_length': len(all_text),
                'regions_detected': len(regions),
                'column_separators': len(v_separators),
                'row_separators': len(h_separators),
                'processing_time': round(processing_time, 2),
                'method': 'histogram_based',
                'status': 'success'
            }
            
            print(f"   Success: {name} - {processing_time:.2f}s - {len(regions)} regions")
            return result
            
        except Exception as e:
            return {
                'filename': os.path.basename(pdf_path),
                'name': '', 'emails': '', 'phones': '', 'linkedin': '', 'github': '',
                'skills': '', 'total_text_length': 0, 'regions_detected': 0,
                'column_separators': 0, 'row_separators': 0,
                'processing_time': 0, 'method': 'histogram_based',
                'status': f'error: {str(e)}'
            }

class BatchHistogramProcessor:
    def __init__(self):
        self.parser = HistogramCVParser()
    
    def process_folder(self, input_folder, output_csv):
        """Process all PDFs in folder and save to CSV"""
        
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist!")
            return
        
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in '{input_folder}'")
            return
        
        print(f"Found {len(pdf_files)} PDF files for histogram-based processing")
        
        # CSV columns
        csv_columns = [
            'Filename', 'Name', 'Emails', 'Phones', 'LinkedIn', 'GitHub',
            'Skills', 'Total_Text_Length', 'Regions_Detected', 
            'Column_Separators', 'Row_Separators', 'Processing_Time_Seconds',
            'Method', 'Status'
        ]
        
        results = []
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_path.name}")
            
            result = self.parser.process_single_cv(str(pdf_path))
            results.append(result)
            
            if 'error' not in result['status']:
                successful += 1
            else:
                failed += 1
        
        # Save to CSV
        try:
            # Ensure output directory exists
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                
                for result in results:
                    # Map result keys to CSV columns
                    csv_row = {
                        'Filename': result['filename'],
                        'Name': result['name'],
                        'Emails': result['emails'],
                        'Phones': result['phones'],
                        'LinkedIn': result['linkedin'],
                        'GitHub': result['github'],
                        'Skills': result['skills'],
                        'Total_Text_Length': result['total_text_length'],
                        'Regions_Detected': result['regions_detected'],
                        'Column_Separators': result['column_separators'],
                        'Row_Separators': result['row_separators'],
                        'Processing_Time_Seconds': result['processing_time'],
                        'Method': result['method'],
                        'Status': result['status']
                    }
                    writer.writerow(csv_row)
            
            print(f"\n=== HISTOGRAM-BASED BATCH PROCESSING COMPLETE ===")
            print(f"Total files processed: {len(pdf_files)}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print(f"Success rate: {(successful/len(pdf_files)*100):.1f}%")
            print(f"CSV saved to: {output_csv}")
            
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")

def main():
    print("=== Enhanced Histogram-Based CV Parser ===")
    print("Features:")
    print(" 2D Histogram analysis for column/row detection")
    print(" Multi-column layout support")
    print(" Intelligent region extraction")
    print(" Contact info and skills extraction")
    print(" Batch processing")
    print(" CSV output with detailed metrics")
    print("=" * 50)
    
    # Configuration
    input_folder = "F:/Cogntix/Unblit/AI/SampleCVs/"
    output_csv = "F:/Cogntix/Unblit/AI/SampleCVs/histogram_extracted_results.csv"
    
    # Initialize processor
    processor = BatchHistogramProcessor()
    
    # Process all files
    print(f"\nInput folder: {input_folder}")
    print(f"Output CSV: {output_csv}")
    
    response = input("\nStart histogram-based processing? (y/n): ")
    if response.lower() != 'y':
        print("Processing cancelled.")
        return
    
    start_time = time.time()
    processor.process_folder(input_folder, output_csv)
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
