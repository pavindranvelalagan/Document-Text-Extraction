# enhanced_document_yolo_extractor.py - COMPLETE WORKING VERSION
import os
import cv2
import numpy as np
from ultralytics import YOLO
import pdfplumber
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import requests
import pandas as pd

class DocumentYOLOExtractor:
    def __init__(self):
        """Initialize with document-specific YOLO model"""
        print("Initializing Document Layout YOLO Extractor")
        print("=" * 60)
        
        # Load the best available model
        self.model = self.load_document_model()
        
    def load_document_model(self):
        """Load appropriate document layout model"""
        
        # Try YOLOv8 segmentation (best available option)
        try:
            print("Loading YOLOv8 segmentation model...")
            # v1.0: model = YOLO("yolov8n-seg.pt")  # Segmentation version
            model = YOLO("nakamura196/yolov8-ndl-layout")
            print("Loaded YOLOv8 segmentation model")
            return model
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
    
    def pdf_to_images(self, pdf_path, dpi=200):
        """
        Convert PDF pages to images for YOLO processing
        
        WHY: YOLO works on images, not PDFs
        HOW: Use pdfplumber to convert each page to high-resolution image
        """
        print(f"\nConverting PDF to images: {pdf_path}")
        print("=" * 50)
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        images = []
        page_info = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"Total pages: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages):
                    print(f"Processing page {page_num + 1}/{total_pages}")
                    
                    # Convert page to image
                    img = page.to_image(resolution=dpi)
                    
                    # Convert PIL image to OpenCV format (BGR)
                    opencv_img = cv2.cvtColor(np.array(img.original), cv2.COLOR_RGB2BGR)
                    
                    images.append(opencv_img)
                    page_info.append({
                        'page_num': page_num,
                        'width_points': page.width,
                        'height_points': page.height,
                        'width_pixels': opencv_img.shape[1],
                        'height_pixels': opencv_img.shape[0],
                        'dpi': dpi
                    })
                    
                    print(f"Page {page_num + 1}: {opencv_img.shape[1]}x{opencv_img.shape[0]} pixels")
        
        except Exception as e:
            print(f"Error converting PDF: {e}")
            raise
        
        print(f"Successfully converted {len(images)} pages to images")
        return images, page_info
    
    def detect_with_custom_classes(self, image, confidence_threshold=0.2):
        """Enhanced detection with document-aware post-processing"""
        print(f"Running document-aware detection...")
        
        # Run base YOLO detection
        results = self.model(image, conf=confidence_threshold, verbose=False)
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # If we get a large detection covering most of the page, split it
                if self.is_large_detection(box, image.shape):
                    print(f"Large '{class_name}' detection found - splitting into sections")
                    sub_detections = self.split_large_detection(box, image)
                    detections.extend(sub_detections)
                else:
                    detection = {
                        'bbox': box,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.map_to_document_class(class_name),
                        'area': (box[2] - box[0]) * (box[3] - box[1])
                    }
                    detections.append(detection)
        
        # If no detections or only very few, create default sections
        if len(detections) == 0:
            print("No detections found - creating default sections")
            detections = self.create_default_sections(image.shape)
        
        print(f"Found {len(detections)} document regions")
        
        # Sort by reading order (top-to-bottom, left-to-right)
        detections.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        return detections
    
    def is_large_detection(self, bbox, img_shape):
        """Check if detection covers too much of the page"""
        x1, y1, x2, y2 = bbox
        detection_area = (x2 - x1) * (y2 - y1)
        total_area = img_shape[0] * img_shape[1]
        
        # If detection covers more than 70% of page, it's too large
        coverage = detection_area / total_area
        print(f"Detection covers {coverage*100:.1f}% of page")
        return coverage > 0.7
    
    def split_large_detection(self, large_bbox, image):
        """Split large detection into logical sections using computer vision"""
        print("Splitting large detection into sections...")
        
        x1, y1, x2, y2 = large_bbox
        
        # Extract the region
        roi = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Find horizontal separators (gaps between sections)
        horizontal_profile = np.mean(gray, axis=1)
        
        # Smooth the profile to reduce noise
        kernel_size = max(3, len(horizontal_profile) // 100)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(horizontal_profile, kernel, mode='same')
        
        # Find gaps (low intensity areas)
        threshold = np.mean(smoothed) * 0.6  # 60% of mean
        gap_positions = np.where(smoothed < threshold)[0]
        
        # Group consecutive gaps
        section_boundaries = [0]  # Start of first section
        
        if len(gap_positions) > 10:  # Only if we have significant gaps
            gap_groups = []
            if len(gap_positions) > 0:
                current_group = [gap_positions[0]]
                
                for pos in gap_positions[1:]:
                    if pos == current_group[-1] + 1:
                        current_group.append(pos)
                    else:
                        if len(current_group) > 8:  # Significant gap
                            gap_center = (current_group[0] + current_group[-1]) // 2
                            section_boundaries.append(gap_center)
                        current_group = [pos]
                
                # Handle last group
                if len(current_group) > 8:
                    gap_center = (current_group[0] + current_group[-1]) // 2
                    section_boundaries.append(gap_center)
        
        section_boundaries.append(len(smoothed))  # End of last section
        
        # Create detections for each section
        sub_detections = []
        
        for i in range(len(section_boundaries) - 1):
            sec_y1 = y1 + section_boundaries[i]
            sec_y2 = y1 + section_boundaries[i + 1]
            
            # Skip very small sections
            if sec_y2 - sec_y1 < 30:
                continue
            
            # Classify section based on position
            section_class = self.classify_section(i, len(section_boundaries) - 1)
            
            sub_detection = {
                'bbox': np.array([x1, sec_y1, x2, sec_y2]),
                'confidence': 0.8,
                'class_id': i,
                'class_name': section_class,
                'area': (x2 - x1) * (sec_y2 - sec_y1)
            }
            
            sub_detections.append(sub_detection)
        
        print(f"Split into {len(sub_detections)} sections")
        return sub_detections
    
    def create_default_sections(self, img_shape):
        """Create default sections when YOLO finds nothing"""
        height, width = img_shape[:2]
        
        # Create 4 default sections
        sections = [
            {'name': 'header', 'y_start': 0, 'y_end': height * 0.25},
            {'name': 'summary', 'y_start': height * 0.25, 'y_end': height * 0.5},
            {'name': 'experience', 'y_start': height * 0.5, 'y_end': height * 0.75},
            {'name': 'education', 'y_start': height * 0.75, 'y_end': height}
        ]
        
        detections = []
        for i, section in enumerate(sections):
            detection = {
                'bbox': np.array([0, section['y_start'], width, section['y_end']]),
                'confidence': 0.5,
                'class_id': i,
                'class_name': section['name'],
                'area': width * (section['y_end'] - section['y_start'])
            }
            detections.append(detection)
        
        return detections
    
    def classify_section(self, section_index, total_sections):
        """Classify document section based on position"""
        if section_index == 0:
            return "header"
        elif section_index == 1:
            return "summary"  
        elif section_index < total_sections * 0.6:
            return "experience"
        elif section_index < total_sections * 0.8:
            return "education"
        else:
            return "skills"
    
    def map_to_document_class(self, original_class):
        """Map general YOLO classes to document-specific classes"""
        mapping = {
            "book": "text",
            "person": "header",
            "text": "text",
        }
        return mapping.get(original_class, "text")
    
    def extract_text_from_regions(self, pdf_path, detections, page_info, page_num):
        """Extract text from each detected region"""
        print(f"Extracting text from {len(detections)} regions")
        
        if page_num >= len(page_info):
            print(f"Page info not available for page {page_num}")
            return []
        
        page_data = page_info[page_num]
        
        # Scale factors to convert pixels back to PDF points
        x_scale = page_data['width_points'] / page_data['width_pixels']
        y_scale = page_data['height_points'] / page_data['height_pixels']
        
        region_texts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num]
                
                for i, detection in enumerate(detections):
                    bbox_pixels = detection['bbox']
                    x1, y1, x2, y2 = bbox_pixels
                    
                    # Convert to PDF coordinates
                    pdf_x1 = x1 * x_scale
                    pdf_y1 = y1 * y_scale
                    pdf_x2 = x2 * x_scale
                    pdf_y2 = y2 * y_scale
                    
                    # Ensure coordinates are within page bounds
                    pdf_x1 = max(0, min(pdf_x1, page.width))
                    pdf_x2 = max(0, min(pdf_x2, page.width))
                    pdf_y1 = max(0, min(pdf_y1, page.height))
                    pdf_y2 = max(0, min(pdf_y2, page.height))
                    
                    pdf_bbox = (pdf_x1, pdf_y1, pdf_x2, pdf_y2)
                    
                    try:
                        # Extract text from this region
                        cropped_page = page.crop(pdf_bbox)
                        text = cropped_page.extract_text() or ""
                        text = text.strip()
                        
                        region_info = {
                            'region_id': i,
                            'page': page_num,
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'bbox_pixels': bbox_pixels,
                            'bbox_pdf': pdf_bbox,
                            'text': text,
                            'char_count': len(text),
                            'area': detection['area']
                        }
                        
                        region_texts.append(region_info)
                        
                        print(f"Region {i} ({detection['class_name']}): {len(text)} chars")
                        
                    except Exception as e:
                        print(f"Error extracting region {i}: {e}")
                        
        except Exception as e:
            print(f"Error in text extraction: {e}")
        
        print(f"Successfully extracted text from {len(region_texts)} regions")
        return region_texts
    
    def visualize_enhanced(self, image, detections, page_num, save_path):
        """Enhanced visualization with section labels"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Different colors for different section types
        color_map = {
            'header': 'red',
            'summary': 'orange', 
            'experience': 'green',
            'education': 'blue',
            'skills': 'purple',
            'text': 'gray'
        }
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            color = color_map.get(class_name, 'gray')
            
            # Draw bounding box
            rect = Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add label
            label = f"{class_name.upper()}\n({confidence:.2f})"
            ax.text(x1 + 5, y1 + 20, label, color=color, fontsize=9, 
                   weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8))
        
        ax.set_title(f'Page {page_num} - Enhanced Document Detection\n{len(detections)} Sections Found')
        ax.axis('on')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved enhanced visualization: {save_path}")
    
    def save_enhanced_results(self, extractions, output_dir, pdf_path):
        """Save extraction results in multiple formats"""
        
        print(f"\nSaving results to {output_dir}")
        print("=" * 50)
        
        if not extractions:
            print("No extractions to save")
            return
        
        # Save detailed text file
        txt_path = os.path.join(output_dir, 'enhanced_extracted_text.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED YOLO-BASED CV TEXT EXTRACTION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Source: {os.path.basename(pdf_path)}\n")
            f.write(f"Total regions: {len(extractions)}\n")
            f.write("=" * 60 + "\n\n")
            
            current_page = -1
            for extraction in extractions:
                if extraction['page'] != current_page:
                    current_page = extraction['page']
                    f.write(f"\n{'='*20} PAGE {current_page + 1} {'='*20}\n\n")
                
                f.write(f"REGION {extraction['region_id']} - {extraction['class_name'].upper()}\n")
                f.write(f"Confidence: {extraction['confidence']:.3f}\n")
                f.write("-" * 40 + "\n")
                f.write(extraction['text'] + "\n\n")
        
        # Save combined text in reading order
        combined_path = os.path.join(output_dir, 'combined_enhanced_text.txt')
        with open(combined_path, 'w', encoding='utf-8') as f:
            for extraction in extractions:
                f.write(extraction['text'] + "\n")
        
        print(f"Saved detailed results: {txt_path}")
        print(f"Saved combined text: {combined_path}")
        
        # Print summary statistics
        total_chars = sum(e['char_count'] for e in extractions)
        pages = len(set(e['page'] for e in extractions))
        classes = set(e['class_name'] for e in extractions)
        
        print(f"\nEXTRACTION SUMMARY:")
        print(f"   Pages processed: {pages}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Detected classes: {', '.join(classes)}")
        print(f"   Average confidence: {np.mean([e['confidence'] for e in extractions]):.3f}")
    
    def process_cv_enhanced(self, pdf_path, output_dir="enhanced_yolo_output"):
        """Enhanced processing with better section detection"""
        
        print("ENHANCED DOCUMENT YOLO PROCESSING")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images
        images, page_info = self.pdf_to_images(pdf_path)
        
        all_extractions = []
        
        # Process each page
        for page_num, image in enumerate(images):
            print(f"\n{'='*20} PAGE {page_num + 1} {'='*20}")
            
            # Enhanced detection with splitting
            detections = self.detect_with_custom_classes(image, confidence_threshold=0.2)
            
            # Visualize
            viz_path = os.path.join(output_dir, f'enhanced_detections_page_{page_num + 1}.png')
            self.visualize_enhanced(image, detections, page_num + 1, viz_path)
            
            # Extract text
            region_texts = self.extract_text_from_regions(pdf_path, detections, page_info, page_num)
            
            all_extractions.extend(region_texts)
        
        # Save results
        self.save_enhanced_results(all_extractions, output_dir, pdf_path)
        
        return all_extractions

def main():
    print("ENHANCED DOCUMENT YOLO EXTRACTOR")
    print("=" * 60)
    print("Splits large detections into logical sections")
    print("Document-aware classification")  
    print("Better section detection")
    print("=" * 60)
    
    pdf_path = input("Enter path to your CV PDF: ").strip()
    if not pdf_path:
        pdf_path = "F:\\Cogntix\\Unblit\\AI\\SampleCVs\\Sample1.pdf"
        print(f"Using default: {pdf_path}")
    
    try:
        extractor = DocumentYOLOExtractor()
        extractions = extractor.process_cv_enhanced(pdf_path)
        
        if extractions:
            print(f"\n SUCCESS! Found {len(extractions)} document sections")
            print("Check 'enhanced_yolo_output' for results:")
            print("   • enhanced_extracted_text.txt - Detailed sections")
            print("   • combined_enhanced_text.txt - All text combined")
            print("   • enhanced_detections_page_X.png - Visual verification")
        else:
            print(" No sections detected")
            
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
