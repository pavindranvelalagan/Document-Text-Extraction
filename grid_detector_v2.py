# enhanced_grid_detector_fixed.py - Multi-page, Non-blocking, Error-free
import pdfplumber
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def clip_bbox(bbox, max_width, max_height):
    """Clip bounding box to stay within boundaries"""
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(x0, max_width))
    x1 = max(0, min(x1, max_width))  
    y0 = max(0, min(y0, max_height))
    y1 = max(0, min(y1, max_height))
    
    # Ensure valid rectangle
    if x0 >= x1: x1 = x0 + 1
    if y0 >= y1: y1 = y0 + 1
    
    return (x0, y0, x1, y1)

def compute_2d_histograms(cv_img, smooth_kernel=5):
    """Compute both vertical AND horizontal histograms"""
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Smooth to reduce noise
    smoothed = cv2.blur(binary, (smooth_kernel, smooth_kernel))
    
    # Both directions
    vertical_hist = np.sum(smoothed, axis=0)    # Column density
    horizontal_hist = np.sum(smoothed, axis=1)  # Row density
    
    return vertical_hist, horizontal_hist, binary

def find_separators_1d(histogram, min_gap_ratio=0.03, threshold_ratio=0.08, margin_ratio=0.05):
    """Find separators in one direction with proper boundary handling"""
    
    total_length = len(histogram)
    max_density = np.max(histogram)
    
    if max_density == 0:
        return []
    
    # Parameters
    threshold = max_density * threshold_ratio
    min_gap_width = int(total_length * min_gap_ratio)
    margin_ignore = int(total_length * margin_ratio)
    
    # Find gaps below threshold
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
            # End current gap
            width = prev - start + 1
            if width >= min_gap_width and start > margin_ignore and prev < (total_length - margin_ignore):
                center = min((start + prev) // 2, total_length - 1)  # Ensure within bounds
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
    if width >= min_gap_width and start > margin_ignore and prev < (total_length - margin_ignore):
        center = min((start + prev) // 2, total_length - 1)  # Ensure within bounds
        gaps.append({
            'start': int(start),
            'end': int(prev),
            'center': int(center), 
            'width': int(width)
        })
    
    return gaps

def create_text_regions_fixed(v_separators, h_separators, img_width, img_height, page_width, page_height):
    """Create rectangular text regions with proper clipping"""
    
    # Create boundaries
    x_boundaries = [0]
    for sep in v_separators:
        x_boundaries.append(min(sep['center'], img_width - 1))
    x_boundaries.append(img_width)
    x_boundaries = sorted(list(set(x_boundaries)))  # Remove duplicates and sort
    
    y_boundaries = [0]
    for sep in h_separators:
        y_boundaries.append(min(sep['center'], img_height - 1))
    y_boundaries.append(img_height)
    y_boundaries = sorted(list(set(y_boundaries)))  # Remove duplicates and sort
    
    # Scale factors
    x_scale = page_width / img_width
    y_scale = page_height / img_height
    
    regions = []
    region_id = 0
    
    for i in range(len(y_boundaries) - 1):  # Rows
        for j in range(len(x_boundaries) - 1):  # Columns
            
            x1, x2 = x_boundaries[j], x_boundaries[j + 1]
            y1, y2 = y_boundaries[i], y_boundaries[i + 1]
            
            # Skip very small regions
            if (x2 - x1) < 50 or (y2 - y1) < 30:
                continue
            
            # Clip to image boundaries
            x1, y1, x2, y2 = clip_bbox((x1, y1, x2, y2), img_width, img_height)
            
            # Convert to PDF coordinates with clipping
            pdf_x1 = x1 * x_scale
            pdf_y1 = y1 * y_scale
            pdf_x2 = x2 * x_scale
            pdf_y2 = y2 * y_scale
            
            # Clip to page boundaries
            pdf_bbox = clip_bbox((pdf_x1, pdf_y1, pdf_x2, pdf_y2), page_width, page_height)
            
            # Ensure valid region
            if pdf_bbox[2] <= pdf_bbox[0] or pdf_bbox[3] <= pdf_bbox[1]:
                continue
                
            regions.append({
                'id': region_id,
                'bbox_pixels': (x1, y1, x2, y2),
                'bbox_pdf': pdf_bbox,
                'width': x2 - x1,
                'height': y2 - y1,
                'row': i,
                'col': j
            })
            region_id += 1
    
    return regions

def visualize_grid_nonblocking(cv_img, regions, v_separators, h_separators, page_num):
    """NON-BLOCKING visualization - saves without showing"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    
    # Show original image
    ax.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # Draw separators
    for sep in v_separators:
        ax.axvline(x=sep['center'], color='red', linewidth=2, alpha=0.7)
    
    for sep in h_separators:
        ax.axhline(y=sep['center'], color='blue', linewidth=2, alpha=0.7)
    
    # Draw region boxes
    for region in regions:
        x1, y1, x2, y2 = region['bbox_pixels']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='green', 
                        facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add region labels
        ax.text(x1 + 5, y1 + 20, f"R{region['id']}", 
                color='green', fontsize=10, weight='bold')
    
    ax.set_title(f'Page {page_num} - 2D Grid Detection')
    ax.axis('on')
    
    # NON-BLOCKING: Save without showing
    filename = f'grid_analysis_page_{page_num}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Important: Close to free memory
    
    print(f"✅ Saved {filename} (non-blocking)")

def extract_region_texts_safe(pdf_path, regions, page_num):
    """Safe text extraction with proper error handling"""
    
    region_texts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                print(f"❌ Page {page_num} doesn't exist")
                return []
                
            page = pdf.pages[page_num]
            
            for region in regions:
                try:
                    # Use pre-calculated PDF bbox
                    pdf_bbox = region['bbox_pdf']
                    
                    # Additional safety check
                    if (pdf_bbox[0] >= pdf_bbox[2] or pdf_bbox[1] >= pdf_bbox[3] or
                        pdf_bbox[2] > page.width or pdf_bbox[3] > page.height):
                        print(f"⚠️ Skipping invalid region {region['id']}: {pdf_bbox}")
                        continue
                    
                    cropped_page = page.crop(pdf_bbox)
                    text = cropped_page.extract_text() or ""
                    text = text.strip()
                    
                    region_info = {
                        'region_id': region['id'],
                        'page': page_num,
                        'bbox_pdf': pdf_bbox,
                        'text': text,
                        'char_count': len(text),
                        'row': region['row'],
                        'col': region['col']
                    }
                    
                    region_texts.append(region_info)
                    print(f"📄 Page {page_num}, Region {region['id']}: {len(text)} characters")
                    
                except Exception as e:
                    print(f"❌ Error extracting Page {page_num}, Region {region['id']}: {e}")
                    
    except Exception as e:
        print(f"❌ Error opening PDF for page {page_num}: {e}")
        
    return region_texts

def process_single_page(pdf_path, page_num):
    """Process one page and return extracted regions"""
    
    print(f"\n📄 PROCESSING PAGE {page_num + 1}")
    print("=" * 40)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                print(f"❌ Page {page_num + 1} doesn't exist")
                return []
                
            page = pdf.pages[page_num]
            
            # Create image
            img = page.to_image(resolution=200)
            cv_img = cv2.cvtColor(np.array(img.original), cv2.COLOR_RGB2BGR)
            
            H, W = cv_img.shape[:2]
            
            # Compute histograms
            ### IMPORTANT ###
            #Change the smooth kernel to test - too much smooting might blur out narrow column gaps
            v_hist, h_hist, binary = compute_2d_histograms(cv_img, smooth_kernel=7)
            
            # Find separators
            ### IMPORTANT ###
            # Adjust the ming_gap_rtio and threshold_ratio to test
            v_separators = find_separators_1d(
                v_hist, 
                min_gap_ratio=0.025, 
                threshold_ratio=0.02
                )
            h_separators = find_separators_1d(
                h_hist, 
                min_gap_ratio=0.05, 
                threshold_ratio=0.20
                )
            
            print(f"📍 Vertical separators: {len(v_separators)}")
            print(f"📍 Horizontal separators: {len(h_separators)}")
            
            # Create regions with proper clipping
            regions = create_text_regions_fixed(v_separators, h_separators, W, H, page.width, page.height)
            
            # Non-blocking visualization
            visualize_grid_nonblocking(cv_img, regions, v_separators, h_separators, page_num + 1)
            
            # Extract texts
            region_texts = extract_region_texts_safe(pdf_path, regions, page_num)
            
            return region_texts
            
    except Exception as e:
        print(f"❌ Error processing page {page_num + 1}: {e}")
        return []

def main():
    """MULTI-PAGE processing with all fixes"""
    
    print("🚀 ENHANCED 2D GRID CV EXTRACTION")
    print("=" * 60)
    print("✅ Multi-page processing")
    print("✅ Non-blocking visualization") 
    print("✅ Fixed bounding box errors")
    print("=" * 60)
    
    # Get PDF path
    pdf_path = input("Enter path to your CV PDF: ").strip()
    if not pdf_path:
        pdf_path = "F:\\Cogntix\\Unblit\\AI\\SampleCVs\\Sample60.pdf"
        print(f"Using default: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Get total pages
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"📁 Processing {total_pages} pages from: {pdf_path}")
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return
    
    # Process all pages
    all_regions = []
    
    for page_num in range(total_pages):
        page_regions = process_single_page(pdf_path, page_num)
        all_regions.extend(page_regions)
    
    # Save combined results
    if all_regions:
        with open('extracted_regions_all_pages.txt', 'w', encoding='utf-8') as f:
            f.write("MULTI-PAGE 2D GRID EXTRACTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Group by page
            current_page = -1
            for region in sorted(all_regions, key=lambda x: (x['page'], x['row'], x['col'])):
                if region['page'] != current_page:
                    current_page = region['page']
                    f.write(f"\n{'='*20} PAGE {current_page + 1} {'='*20}\n\n")
                
                f.write(f"REGION {region['region_id']} (Row {region['row']}, Col {region['col']}):\n")
                f.write("-" * 30 + "\n")
                f.write(region['text'] + "\n\n")
        
        print(f"\n✅ COMPLETE! Processed {total_pages} pages")
        print(f"📄 Extracted {len(all_regions)} text regions total")
        print(f"💾 Saved: extracted_regions_all_pages.txt")
        print(f"🖼️  Created grid_analysis_page_X.png for each page")
    else:
        print("❌ No regions extracted from any page")

if __name__ == "__main__":
    main()
