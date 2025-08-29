# enhanced_grid_detector.py - Fixed version with proper PDF handling
import pdfplumber
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def compute_2d_histograms(cv_img, smooth_kernel=5):
    """
    STEP 1: Compute both vertical AND horizontal histograms
    """
    
    print("\n🔍 STEP 1: Computing 2D Histograms")
    print("=" * 50)
    
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Smooth to reduce noise from small elements
    smoothed = cv2.blur(binary, (smooth_kernel, smooth_kernel))
    
    # Your approach: BOTH directions
    vertical_hist = np.sum(smoothed, axis=0)    # Sum along rows → column density
    horizontal_hist = np.sum(smoothed, axis=1)  # Sum along cols → row density
    
    print(f"✅ Vertical histogram: {len(vertical_hist)} points")
    print(f"✅ Horizontal histogram: {len(horizontal_hist)} points")
    
    return vertical_hist, horizontal_hist, binary

def find_separators_1d(histogram, min_gap_ratio=0.03, threshold_ratio=0.08, margin_ratio=0.05):
    """Find separators in one direction (vertical OR horizontal)"""
    
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
    if width >= min_gap_width and start > margin_ignore and prev < (total_length - margin_ignore):
        center = (start + prev) // 2
        gaps.append({
            'start': int(start),
            'end': int(prev),
            'center': int(center), 
            'width': int(width)
        })
    
    return gaps

def create_text_regions(v_separators, h_separators, img_width, img_height):
    """Create rectangular text regions from separators"""
    
    print(f"\n🔍 STEP 2: Creating Text Regions Grid")
    print("=" * 50)
    
    # Create column boundaries (x-coordinates)
    x_boundaries = [0]  # Start at left edge
    for sep in v_separators:
        x_boundaries.append(sep['center'])
    x_boundaries.append(img_width)  # End at right edge
    x_boundaries.sort()
    
    # Create row boundaries (y-coordinates)  
    y_boundaries = [0]  # Start at top edge
    for sep in h_separators:
        y_boundaries.append(sep['center'])
    y_boundaries.append(img_height)  # End at bottom edge
    y_boundaries.sort()
    
    print(f"📏 Column boundaries: {x_boundaries}")
    print(f"📏 Row boundaries: {y_boundaries}")
    
    # Create rectangular regions from grid intersections
    regions = []
    region_id = 0
    
    for i in range(len(y_boundaries) - 1):  # Rows
        for j in range(len(x_boundaries) - 1):  # Columns
            
            x1, x2 = x_boundaries[j], x_boundaries[j + 1]
            y1, y2 = y_boundaries[i], y_boundaries[i + 1]
            
            # Skip very small regions
            if (x2 - x1) < 50 or (y2 - y1) < 30:
                continue
                
            regions.append({
                'id': region_id,
                'bbox_pixels': (x1, y1, x2, y2),
                'width': x2 - x1,
                'height': y2 - y1,
                'row': i,
                'col': j
            })
            region_id += 1
    
    print(f"✅ Created {len(regions)} text regions")
    
    return regions

def visualize_grid(cv_img, regions, v_separators, h_separators):
    """Visualize the detected grid for verification"""
    
    print(f"\n🔍 STEP 3: Visualizing Grid Layout")
    print("=" * 50)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    
    # Show original image
    ax.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    
    # Draw vertical separators (red lines)
    for sep in v_separators:
        ax.axvline(x=sep['center'], color='red', linewidth=2, alpha=0.7)
    
    # Draw horizontal separators (blue lines)  
    for sep in h_separators:
        ax.axhline(y=sep['center'], color='blue', linewidth=2, alpha=0.7)
    
    # Draw text region boxes (green rectangles)
    for region in regions:
        x1, y1, x2, y2 = region['bbox_pixels']
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='green', 
                        facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add region ID labels
        ax.text(x1 + 5, y1 + 20, f"R{region['id']}", 
                color='green', fontsize=10, weight='bold')
    
    ax.set_title('2D Grid Detection: Red=Columns, Blue=Rows, Green=Text Regions')
    ax.axis('on')
    
    plt.tight_layout()
    plt.savefig('grid_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved grid visualization: grid_analysis.png")

def extract_region_texts(pdf_path, regions, page_width, page_height, img_width, img_height):
    """Extract text from each region"""
    
    print(f"\n🔍 STEP 4: Extracting Text from {len(regions)} Regions")  
    print("=" * 50)
    
    # Scale factors to convert pixels to PDF points
    x_scale = page_width / img_width
    y_scale = page_height / img_height
    
    region_texts = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            
            for region in regions:
                x1, y1, x2, y2 = region['bbox_pixels']
                
                # Convert pixel coordinates to PDF points
                pdf_bbox = (
                    x1 * x_scale,
                    y1 * y_scale, 
                    x2 * x_scale,
                    y2 * y_scale
                )
                
                # Extract text from this region
                try:
                    cropped_page = page.crop(pdf_bbox)
                    text = cropped_page.extract_text() or ""
                    text = text.strip()
                    
                    region_info = {
                        'region_id': region['id'],
                        'bbox_pdf': pdf_bbox,
                        'text': text,
                        'char_count': len(text),
                        'row': region['row'],
                        'col': region['col']
                    }
                    
                    region_texts.append(region_info)
                    
                    print(f"📄 Region {region['id']}: {len(text)} characters")
                    
                except Exception as e:
                    print(f"❌ Error extracting region {region['id']}: {e}")
                    
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return []
    
    return region_texts

def main():
    """Main function implementing your 2D grid approach"""
    
    print("🚀 2D GRID-BASED CV EXTRACTION")
    print("=" * 60)
    print("YOUR APPROACH: Vertical + Horizontal histogram analysis")
    print("GOAL: Handle complex layouts (2-col top, 3-col bottom, etc.)")
    print("=" * 60)
    
    # Get PDF path
    pdf_path = input("Enter path to your CV PDF: ").strip()
    if not pdf_path:
        pdf_path = "F:\\Cogntix\\Unblit\\AI\\SampleCVs\\Sample60.pdf"
        print(f"Using default path: {pdf_path}")
    
    # Verify file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    print(f"📁 Processing: {pdf_path}")
    
    # FIXED: Proper PDF loading with error handling
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                print("❌ PDF has no pages")
                return
                
            page = pdf.pages[0]  # Get first page
            print(f"✅ Successfully loaded page: {page.width} x {page.height} points")
            
            # Create image from page
            img = page.to_image(resolution=200)
            cv_img = cv2.cvtColor(np.array(img.original), cv2.COLOR_RGB2BGR)
            
            H, W = cv_img.shape[:2]
            print(f"📏 Image size: {W} × {H} pixels")
            
            # Your 2D approach
            v_hist, h_hist, binary = compute_2d_histograms(cv_img, smooth_kernel=7)
            
            # Find separators in both directions
            v_separators = find_separators_1d(v_hist, min_gap_ratio=0.05, threshold_ratio=0.06)
            h_separators = find_separators_1d(h_hist, min_gap_ratio=0.02, threshold_ratio=0.08)
            
            print(f"📍 Vertical separators (columns): {len(v_separators)}")
            print(f"📍 Horizontal separators (rows): {len(h_separators)}")
            
            # Create text regions from grid
            regions = create_text_regions(v_separators, h_separators, W, H)
            
            # Visualize the grid
            visualize_grid(cv_img, regions, v_separators, h_separators)
            
            # Extract text from each region
            region_texts = extract_region_texts(pdf_path, regions, page.width, page.height, W, H)
            
            # Save results
            with open('extracted_regions.txt', 'w', encoding='utf-8') as f:
                f.write("2D GRID EXTRACTION RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                # Sort regions by row then column for proper reading order
                region_texts.sort(key=lambda x: (x['row'], x['col']))
                
                for region in region_texts:
                    f.write(f"REGION {region['region_id']} (Row {region['row']}, Col {region['col']}):\n")
                    f.write("-" * 30 + "\n")
                    f.write(region['text'] + "\n\n")
            
            print("✅ Saved extracted_regions.txt with all region texts")
            print(f"🎯 RESULT: Detected {len(regions)} text regions using your 2D approach!")
            
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        print("🔧 Please check:")
        print("   - File path is correct")
        print("   - PDF is not corrupted")
        print("   - File is not password-protected")

if __name__ == "__main__":
    main()
