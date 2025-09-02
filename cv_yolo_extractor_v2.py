# enhanced_cv_processor.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
import pdfplumber
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from scipy import ndimage

class EnhancedCVProcessor:
    def __init__(self, model_name="yolov8n-seg.pt"):
        """Initialize with document-specific model"""
        print(f"Loading model: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            print(f"Model loaded successfully")
            print(f"Available classes: {list(self.model.names.values())}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
    
    def pdf_to_images(self, pdf_path, dpi=200):
        """Convert PDF pages to high-resolution images"""
        print(f"Converting PDF: {pdf_path}")
        
        images = []
        page_info = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Convert to image
                img = page.to_image(resolution=dpi)
                cv_img = cv2.cvtColor(np.array(img.original), cv2.COLOR_RGB2BGR)
                
                images.append(cv_img)
                page_info.append({
                    'page_num': page_num,
                    'width_points': page.width,
                    'height_points': page.height,
                    'width_pixels': cv_img.shape[1],
                    'height_pixels': cv_img.shape[0]
                })
                
                print(f"Page {page_num + 1}: {cv_img.shape[1]}x{cv_img.shape[0]} pixels")
        
        return images, page_info
    
    def detect_with_segmentation(self, image, confidence_threshold=0.25):
        """Enhanced detection using segmentation masks"""
        print(f"Running segmentation detection (conf={confidence_threshold})")
        
        # Run YOLO inference
        results = self.model(image, conf=confidence_threshold, verbose=False)
        
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Check if segmentation masks are available
            masks = results[0].masks if hasattr(results[0], 'masks') and results[0].masks is not None else None
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Get mask if available
                mask = masks.xy[i] if masks is not None else None
                
                detection = {
                    'bbox': box,
                    'mask': mask,  # Polygon mask for better precision
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.classify_cv_section(box, image.shape, class_name),
                    'area': (box[2] - box[0]) * (box[3] - box[1])
                }
                
                detections.append(detection)
        
        # Apply post-processing
        detections = self.post_process_detections(detections, image)
        
        print(f"Found {len(detections)} processed regions")
        return detections
    
    def classify_cv_section(self, bbox, img_shape, original_class):
        """Classify CV sections based on position and context"""
        x1, y1, x2, y2 = bbox
        height, width = img_shape[:2]
        
        # Calculate relative position
        center_y = (y1 + y2) / 2
        relative_y = center_y / height
        
        # Size-based classification
        box_height = y2 - y1
        relative_height = box_height / height
        
        # Position-based classification for CVs
        if relative_y < 0.2:
            return "header"
        elif relative_y < 0.35 and relative_height < 0.15:
            return "summary"
        elif relative_y < 0.7:
            if relative_height > 0.25:
                return "experience"  # Large section = likely experience
            else:
                return "education"   # Smaller section = likely education
        else:
            return "skills"
    
    def post_process_detections(self, detections, image):
        """Advanced post-processing to improve detection quality"""
        
        if not detections:
            return self.create_fallback_sections(image.shape)
        
        # Step 1: Split oversized detections
        processed = []
        for detection in detections:
            if self.is_oversized(detection['bbox'], image.shape):
                sub_detections = self.split_detection(detection, image)
                processed.extend(sub_detections)
            else:
                processed.append(detection)
        
        # Step 2: Merge nearby small detections
        processed = self.merge_nearby_detections(processed)
        
        # Step 3: Sort by reading order
        processed.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))  # Top-to-bottom, left-to-right
        
        # Step 4: Re-assign IDs and final classification
        for i, detection in enumerate(processed):
            detection['region_id'] = i
            detection['final_class'] = self.refine_classification(detection, processed)
        
        return processed
    
    def is_oversized(self, bbox, img_shape):
        """Check if detection covers too much area"""
        x1, y1, x2, y2 = bbox
        detection_area = (x2 - x1) * (y2 - y1)
        total_area = img_shape[0] * img_shape[1]
        coverage = detection_area / total_area
        
        return coverage > 0.6  # More than 60% = too large
    
    def split_detection(self, detection, image):
        """Split large detection using histogram analysis"""
        print(f"🔪 Splitting large detection...")
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Extract region of interest
        roi = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Convert to grayscale and find horizontal gaps
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        horizontal_profile = np.mean(gray, axis=1)
        
        # Smooth profile
        smoothed = ndimage.gaussian_filter1d(horizontal_profile, sigma=2)
        
        # Find gaps (valleys in the profile)
        threshold = np.mean(smoothed) * 0.6
        gaps = np.where(smoothed < threshold)[0]
        
        # Group consecutive gaps
        section_boundaries = [0]
        if len(gaps) > 10:
            current_group = []
            for gap in gaps:
                if not current_group or gap == current_group[-1] + 1:
                    current_group.append(gap)
                else:
                    if len(current_group) > 8:  # Significant gap
                        section_boundaries.append(current_group[len(current_group)//2])
                    current_group = [gap]
            
            if len(current_group) > 8:
                section_boundaries.append(current_group[len(current_group)//2])
        
        section_boundaries.append(len(smoothed))
        
        # Create sub-detections
        sub_detections = []
        for i in range(len(section_boundaries) - 1):
            sec_y1 = y1 + section_boundaries[i]
            sec_y2 = y1 + section_boundaries[i + 1]
            
            if sec_y2 - sec_y1 > 30:  # Skip tiny sections
                sub_detection = detection.copy()
                sub_detection['bbox'] = np.array([x1, sec_y1, x2, sec_y2])
                sub_detection['area'] = (x2 - x1) * (sec_y2 - sec_y1)
                sub_detections.append(sub_detection)
        
        print(f"Split into {len(sub_detections)} sections")
        return sub_detections
    
    def merge_nearby_detections(self, detections):
        """Merge detections that are too close together"""
        if len(detections) <= 1:
            return detections
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            candidates_to_merge = [det1]
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check if detections are close vertically
                if self.should_merge(det1['bbox'], det2['bbox']):
                    candidates_to_merge.append(det2)
                    used.add(j)
            
            # Merge all candidates
            merged_detection = self.merge_detections(candidates_to_merge)
            merged.append(merged_detection)
            used.add(i)
        
        return merged
    
    def should_merge(self, bbox1, bbox2):
        """Check if two bounding boxes should be merged"""
        _, y1_1, _, y2_1 = bbox1
        _, y1_2, _, y2_2 = bbox2
        
        # Vertical distance between boxes
        gap = min(abs(y1_2 - y2_1), abs(y1_1 - y2_2))
        avg_height = ((y2_1 - y1_1) + (y2_2 - y1_2)) / 2
        
        # Merge if gap is less than 20% of average height
        return gap < avg_height * 0.2
    
    def merge_detections(self, detections_to_merge):
        """Merge multiple detections into one"""
        if len(detections_to_merge) == 1:
            return detections_to_merge[0]
        
        # Find bounding box that encompasses all
        min_x = min(det['bbox'][0] for det in detections_to_merge)
        min_y = min(det['bbox'][1] for det in detections_to_merge)
        max_x = max(det['bbox'][2] for det in detections_to_merge)
        max_y = max(det['bbox'][3] for det in detections_to_merge)
        
        # Take best confidence and most common class
        best_confidence = max(det['confidence'] for det in detections_to_merge)
        classes = [det['class_name'] for det in detections_to_merge]
        most_common_class = max(set(classes), key=classes.count)
        
        merged = detections_to_merge[0].copy()
        merged['bbox'] = np.array([min_x, min_y, max_x, max_y])
        merged['confidence'] = best_confidence
        merged['class_name'] = most_common_class
        merged['area'] = (max_x - min_x) * (max_y - min_y)
        
        return merged
    
    def create_fallback_sections(self, img_shape):
        """Create default sections when no detections found"""
        height, width = img_shape[:2]
        
        sections = [
            {'name': 'header', 'y_ratio': (0, 0.25)},
            {'name': 'summary', 'y_ratio': (0.25, 0.4)},
            {'name': 'experience', 'y_ratio': (0.4, 0.7)},
            {'name': 'education', 'y_ratio': (0.7, 0.9)},
            {'name': 'skills', 'y_ratio': (0.9, 1.0)}
        ]
        
        detections = []
        for i, section in enumerate(sections):
            y1 = int(height * section['y_ratio'][0])
            y2 = int(height * section['y_ratio'][1])
            
            detection = {
                'region_id': i,
                'bbox': np.array([0, y1, width, y2]),
                'mask': None,
                'confidence': 0.5,
                'class_name': section['name'],
                'area': width * (y2 - y1)
            }
            detections.append(detection)
        
        return detections
    
    def refine_classification(self, detection, all_detections):
        """Final pass to refine section classification"""
        # This is where you can add more sophisticated logic
        # For now, return the existing classification
        return detection['class_name']
    
    def visualize_detections(self, image, detections, page_num, save_path):
        """Create visualization with improved layout"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        colors = {
            'header': 'red',
            'summary': 'orange',
            'experience': 'green', 
            'education': 'blue',
            'skills': 'purple',
            'text': 'gray'
        }
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection.get('final_class', detection['class_name'])
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            color = colors.get(class_name, 'gray')
            
            # Draw bounding box
            rect = Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Draw mask if available
            if detection.get('mask') is not None:
                mask = detection['mask']
                if len(mask) > 0:
                    ax.plot(mask[:, 0], mask[:, 1], color=color, linewidth=1, alpha=0.6)
            
            # Add label
            label = f"{class_name.upper()}\n({confidence:.2f})"
            ax.text(x1 + 5, y1 + 20, label, color=color, fontsize=9, 
                   weight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8))
        
        ax.set_title(f'Page {page_num} - Enhanced Detection ({len(detections)} regions)')
        ax.axis('on')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {save_path}")
    
    def extract_text_from_detections(self, pdf_path, detections, page_info, page_num):
        """Extract text from detected regions"""
        print(f"Extracting text from {len(detections)} regions")
        
        page_data = page_info[page_num]
        x_scale = page_data['width_points'] / page_data['width_pixels']
        y_scale = page_data['height_points'] / page_data['height_pixels']
        
        region_texts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            
            for detection in detections:
                bbox_pixels = detection['bbox']
                x1, y1, x2, y2 = bbox_pixels
                
                # Convert to PDF coordinates
                pdf_x1 = max(0, min(x1 * x_scale, page.width))
                pdf_y1 = max(0, min(y1 * y_scale, page.height))
                pdf_x2 = max(0, min(x2 * x_scale, page.width))
                pdf_y2 = max(0, min(y2 * y_scale, page.height))
                
                pdf_bbox = (pdf_x1, pdf_y1, pdf_x2, pdf_y2)
                
                try:
                    cropped_page = page.crop(pdf_bbox)
                    text = cropped_page.extract_text() or ""
                    text = text.strip()
                    
                    region_info = {
                        'region_id': detection.get('region_id', 0),
                        'page': page_num,
                        'class_name': detection.get('final_class', detection['class_name']),
                        'confidence': detection['confidence'],
                        'bbox_pixels': bbox_pixels,
                        'bbox_pdf': pdf_bbox,
                        'text': text,
                        'char_count': len(text),
                        'area': detection['area']
                    }
                    
                    region_texts.append(region_info)
                    
                    print(f"   Region {detection.get('region_id', 0)} ({detection['class_name']}): {len(text)} chars")
                    
                except Exception as e:
                    print(f"   Error extracting region: {e}")
        
        return region_texts
    
    def save_results(self, all_extractions, output_dir, pdf_path):
        """Save extraction results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Detailed results
        with open(os.path.join(output_dir, 'enhanced_extractions.txt'), 'w', encoding='utf-8') as f:
            f.write("ENHANCED CV TEXT EXTRACTION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Source: {os.path.basename(pdf_path)}\n")
            f.write(f"Total regions: {len(all_extractions)}\n")
            f.write("=" * 60 + "\n\n")
            
            current_page = -1
            for extraction in all_extractions:
                if extraction['page'] != current_page:
                    current_page = extraction['page']
                    f.write(f"\n{'='*20} PAGE {current_page + 1} {'='*20}\n\n")
                
                f.write(f"REGION {extraction['region_id']} - {extraction['class_name'].upper()}\n")
                f.write(f"Confidence: {extraction['confidence']:.3f}\n")
                f.write("-" * 40 + "\n")
                f.write(extraction['text'] + "\n\n")
        
        # Combined text
        with open(os.path.join(output_dir, 'combined_enhanced.txt'), 'w', encoding='utf-8') as f:
            for extraction in all_extractions:
                f.write(extraction['text'] + "\n")
        
        print(f"Results saved to {output_dir}")
    
    def process_cv_complete(self, pdf_path, output_dir="enhanced_output"):
        """Complete CV processing pipeline"""
        print("ENHANCED CV PROCESSING PIPELINE")
        print("=" * 60)
        
        # Convert PDF to images
        images, page_info = self.pdf_to_images(pdf_path)
        
        all_extractions = []
        
        # Process each page
        for page_num, image in enumerate(images):
            print(f"\n{'='*20} PAGE {page_num + 1} {'='*20}")
            
            # Detect regions with enhancements
            detections = self.detect_with_segmentation(image, confidence_threshold=0.2)
            
            # Visualize results
            viz_path = os.path.join(output_dir, f'enhanced_page_{page_num + 1}.png')
            os.makedirs(output_dir, exist_ok=True)
            self.visualize_detections(image, detections, page_num + 1, viz_path)
            
            # Extract text
            region_texts = self.extract_text_from_detections(pdf_path, detections, page_info, page_num)
            all_extractions.extend(region_texts)
        
        # Save results
        self.save_results(all_extractions, output_dir, pdf_path)
        
        return all_extractions


def main():
    """Main function to test the enhanced processor"""
    print("ENHANCED CV PROCESSOR")
    print("=" * 50)
    
    # Get input
    pdf_path = input("Enter CV PDF path: ").strip()
    if not pdf_path:
        pdf_path = "F:\\Cogntix\\Unblit\\AI\\SampleCVs\\Sample1.pdf"
    
    model_choice = input("Choose model (1: DocLayNet, 2: YOLOv8-seg): ").strip()
    if model_choice == "1":
        model_name = "hf://DILHTWD/yolov8-doclaynet" 
    else:
        model_name = "yolov8n-seg.pt"
    
    try:
        # Initialize processor
        processor = EnhancedCVProcessor(model_name)
        
        # Process CV
        extractions = processor.process_cv_complete(pdf_path, "enhanced_cv_output")
        
        if extractions:
            print(f"\n SUCCESS! Extracted {len(extractions)} regions")
            print("Check 'enhanced_cv_output' folder for results")
            
            # Print summary
            classes = set(e['class_name'] for e in extractions)
            total_chars = sum(e['char_count'] for e in extractions)
            
            print(f"Detected sections: {', '.join(classes)}")
            print(f"Total characters: {total_chars:,}")
        else:
            print("No regions extracted")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
