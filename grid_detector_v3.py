# final_enhanced_grid_detector.py - Complete Fixed Version
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
    
    print("🔍 Computing 2D Histograms...")
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Different smoothing for vertical vs horizontal
    # Less smoothing for vertical (preserve narrow column gaps)
    v_smoothed = cv2.blur(binary, (3, 1))  # Minimal horizontal smoothing
    h_smoothed = cv2.blur(binary, (1, smooth_kernel))  # More vertical smoothing
    
    # Compute histograms
    vertical_hist = np.sum(v_smoothed, axis=0)    # Column density
    horizontal_hist = np.sum(h_smoothed, axis=1)  # Row density
    
    print(f"✅ Vertical histogram: {len(vertical_hist)} points")
    print(f"✅ Horizontal histogram: {len(horizontal_hist)} points")
    
    return vertical_hist, horizontal_hist, binary

def find_separators_1d_debug(histogram, direction, min_gap_ratio=0.03, threshold_ratio=0.08, margin_ratio=0.05):
    """Find separators with debug information"""
    
    total_length = len(histogram)
    max_density = np.max(histogram)
    mean_density = np.mean(histogram)
    
    print(f"\n🔍 DEBUG: {direction} Separator Detection")
    print(f"📊 Max density: {max_density}")
    print(f"📊 Mean density: {mean_density:.1f}")
    
    if max_density == 0:
        print("❌ No text found in histogram")
        return []
    
    # Parameters
    threshold = max_density * threshold_ratio
    min_gap_width = int(total_length * min_gap_ratio)
    margin_ignore = int(total_length * margin_ratio)
    
    print(f"📏 Threshold: {threshold:.1f} ({threshold_ratio*100}% of max)")
    print(f"📏 Min gap width: {min_gap_width} pixels ({min_gap_ratio*100}% of {direction.lower()})")
    print(f"📏 Margin ignore: {margin_ignore} pixels")
    
    # Find gaps below threshold
    below_threshold = np.where(histogram < threshold)[0]
    
    if len(below_threshold) == 0:
        print(f"❌ No pixels below threshold for {direction}")
        return []
    
    print(f"🔍 Found {len(below_threshold)} pixels below threshold ({len(below_threshold)/total_length*100:.1f}%)")
    
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
                center = min((start + prev) // 2, total_length - 1)
                gaps.append({
                    'start': int(start),
                    'end': int(prev), 
                    'center': int(center),
                    'width': int(width)
                })
                print(f"✅ Gap found: {start}-{prev} (width: {width}, center: {center})")
            start = pos
            prev = pos
    
    # Handle last gap
    width = prev - start + 1
    if width >= min_gap_width and start > margin_ignore and prev < (total_length - margin_ignore):
        center = min((start + prev) // 2, total_length - 1)
        gaps.append({
            'start': int(start),
            'end': int(prev),
            'center': int(center), 
            'width': int(width)
        })
        print(f"✅ Last gap: {start}-{prev} (width: {width}, center: {center})")
    
    print(f"🎯 Total {direction.lower()} separators found: {len(gaps)}")
    return gaps

def find_true_gaps_horizontal(horizontal_hist, min_gap_height_ratio=0.04):
    """
    ALTERNATIVE: Find only TRUE gaps (near-zero density areas)
    More restrictive to avoid cutting through text
    """
    
    print(f"\n🔍 TRUE GAP Detection (Horizontal)")
    
    # Only consider areas with essentially zero text (1% of max)
    max_density = np.max(horizontal_hist)
    true_threshold = max_density * 0.01
    min_gap_height = int(len(horizontal_hist) * min_gap_height_ratio)
    
    print(f"📊 True gap threshold: {true_threshold:.1f} (1% of max)")
    print(f"📏 Min gap height: {min_gap_height} pixels")
    
    # Find completely empty regions
    empty_regions = np.where(horizontal_hist <= true_threshold)[0]
    
    if len(empty_regions) == 0:
        print("❌ No true gaps found")
        return []
    
    print(f"🔍 Found {len(empty_regions)} near-empty pixels")
    
    # Group consecutive empty positions
    gaps = []
    start = empty_regions[0]
    prev = empty_regions[0]
    
    for pos in empty_regions[1:]:
        if pos == prev + 1:
            prev = pos
        else:
            # Check if gap is substantial
            gap_height = prev - start + 1
            if gap_height >= min_gap_height:
                center = (start + prev) // 2
                gaps.append({
                    'start': int(start),
                    'end': int(prev),
                    'center': int(center),
                    'width': int(gap_height)
                })
                print(f"✅ True gap: {start}-{prev} (height: {gap_height})")
            
            start = pos
            prev = pos
    
    # Handle last gap
    gap_height = prev - start + 1
    if gap_height >= min_gap_height:
        center = (start + prev) // 2
        gaps.append({
            'start': int(start),
            'end': int(prev),
            'center': int(center),
            'width': int(gap_height)
        })
        print(f"✅ Last true gap: {start}-{prev} (height: {gap_height})")
    
    print(f"🎯 Total true horizontal gaps: {len(gaps)}")
    return gaps

def create_text_regions_fixed(v_separators, h_separators, img_width, img_height, page_width, page_height):
    """Create rectangular text regions with proper clipping"""
    
    print(f"\n🔍 Creating Text Regions Grid")
    print("=" * 40)
    
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
    
    print(f"📏 Column boundaries: {x_boundaries}")
    print(f"📏 Row boundaries: {y_boundaries}")
    
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
    
    print(f"✅ Created {len(regions)} text regions")
    return regions

def visualize_grid_nonblocking(cv_img, regions, v_separators, h_separators, page_num):
    """NON-BLOCKING visualization"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    
    # Show original image
    ax.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # Draw vertical separators (red lines) - should be fewer
    for i, sep in enumerate(v_separators):
        ax.axvline(x=sep['center'], color='red', linewidth=3, alpha=0.8, label=f'Col Sep {i+1}' if i == 0 else "")
    
    # Draw horizontal separators (blue lines) - should be more restrictive
    for i, sep in enumerate(h_separators):
        ax.axhline(y=sep['center'], color='blue', linewidth=2, alpha=0.7, label=f'Row Sep {i+1}' if i == 0 else "")
    
    # Draw region boxes (green rectangles)
    for region in regions:
        x1, y1, x2, y2 = region['bbox_pixels']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='green', 
                        facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add region labels
        ax.text(x1 + 5, y1 + 20, f"R{region['id']}", 
                color='green', fontsize=10, weight='bold')
    
    ax.set_title(f'Page {page_num} - Fixed 2D Grid Detection\nRed=Columns, Blue=Rows, Green=Regions')
    ax.axis('on')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # NON-BLOCKING: Save without showing
    filename = f'grid_analysis_page_{page_num}_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved {filename} (non-blocking)")

def extract_region_texts_safe(pdf_path, regions, page_num):
    """Safe text extraction with proper error handling"""
    
    region_texts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                print(f"❌ Page {page_num + 1} doesn't exist")
                return []
                
            page = pdf.pages[page_num]
            
            for region in regions:
                try:
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
                    print(f"📄 Page {page_num + 1}, Region {region['id']}: {len(text)} chars")
                    
                except Exception as e:
                    print(f"❌ Error extracting Page {page_num + 1}, Region {region['id']}: {e}")
                    
    except Exception as e:
        print(f"❌ Error opening PDF for page {page_num + 1}: {e}")
        
    return region_texts

def process_single_page(pdf_path, page_num, use_true_gaps=True):
    """Process one page with improved parameters"""
    
    print(f"\n📄 PROCESSING PAGE {page_num + 1}")
    print("=" * 50)
    
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
            print(f"📏 Image size: {W} × {H} pixels")
            
            # Compute histograms with different smoothing
            v_hist, h_hist, binary = compute_2d_histograms(cv_img, smooth_kernel=5)
            
            # IMPROVED PARAMETERS:
            
            # Vertical: More sensitive (detect narrow column gaps)
            v_separators = find_separators_1d_debug(v_hist, "Vertical",
                min_gap_ratio=0.025,      # Allow 2.5% width gaps
                threshold_ratio=0.025,    # Very sensitive (2.5% of max)
                margin_ratio=0.03)        # Small margin
            
            # Horizontal: Two options - choose one
            if use_true_gaps:
                # OPTION 1: True gaps only (recommended)
                h_separators = find_true_gaps_horizontal(h_hist, min_gap_height_ratio=0.04)
            else:
                # OPTION 2: Restrictive threshold method
                h_separators = find_separators_1d_debug(h_hist, "Horizontal",
                    min_gap_ratio=0.05,       # Require 5% height gaps
                    threshold_ratio=0.20,     # Only very empty areas (20% of max)
                    margin_ratio=0.05)        # Standard margin
            
            print(f"\n🎯 FINAL RESULTS:")
            print(f"📍 Vertical separators: {len(v_separators)}")
            print(f"📍 Horizontal separators: {len(h_separators)}")
            
            # Create regions
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
    """FINAL Multi-page processing with improved parameters"""
    
    print("🚀 FINAL ENHANCED 2D GRID CV EXTRACTION")
    print("=" * 60)
    print("✅ Improved vertical detection (sensitive)")
    print("✅ Improved horizontal detection (restrictive)")
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
    
    # Ask for gap detection method
    use_true_gaps = input("Use true gap detection for horizontal? (y/n, default=y): ").strip().lower()
    use_true_gaps = use_true_gaps != 'n'
    
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
        page_regions = process_single_page(pdf_path, page_num, use_true_gaps)
        all_regions.extend(page_regions)
    
    # Save combined results
    if all_regions:
        with open('extracted_regions_final.txt', 'w', encoding='utf-8') as f:
            f.write("FINAL MULTI-PAGE 2D GRID EXTRACTION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Method: {'True Gap Detection' if use_true_gaps else 'Threshold Method'}\n")
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
        
        print(f"\n🎉 COMPLETE! Processed {total_pages} pages")
        print(f"📄 Extracted {len(all_regions)} text regions total")
        print(f"💾 Saved: extracted_regions_final.txt")
        print(f"🖼️  Created grid_analysis_page_X_fixed.png for each page")
        
        # Summary statistics
        pages_with_regions = len(set(r['page'] for r in all_regions))
        avg_regions_per_page = len(all_regions) / pages_with_regions if pages_with_regions > 0 else 0
        print(f"📊 Average regions per page: {avg_regions_per_page:.1f}")
        
    else:
        print("❌ No regions extracted from any page")

if __name__ == "__main__":
    main()
