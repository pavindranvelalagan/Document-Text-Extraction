# column_detector.py - Complete Column Detection Script
# Purpose: Analyze PDF CV layout and detect column boundaries

import pdfplumber
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_cv_layout(pdf_path):
    """
    STEP 1: Visual analysis of CV layout
    
    WHY: We need to see the actual layout before writing detection code
    WHAT: Creates a visual preview and shows page structure
    """
    
    print("STEP 1: Analyzing CV Layout")
    print("=" * 50)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]  # First page
            
            # Get page information
            width = page.width
            height = page.height
            
            print(f"Page Size: {width} x {height} points")
            print(f"Aspect Ratio: {width/height:.2f}")
            
            # Create high-resolution image for analysis
            img = page.to_image(resolution=200)  # Higher resolution = better analysis
            
            # Save the image
            img.save("layout_analysis.png")
            print("Saved high-res layout image: layout_analysis.png")
            
            # Convert to OpenCV format for computer vision processing
            # WHY: OpenCV has powerful image analysis tools
            # WHAT: Converts PIL image to numpy array that OpenCV can process
            
            pil_img = img.original
            opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            print(f"Image dimensions: {opencv_img.shape}")
            print("Converted to OpenCV format for analysis")
            
            return page, opencv_img, width, height
            
    except Exception as e:
        print(f"Error in layout analysis: {e}")
        return None, None, None, None

def detect_column_boundaries(opencv_img):
    """
    STEP 2: Find column boundaries using pixel analysis
    
    WHY: Computers can precisely detect whitespace gaps that separate columns
    HOW: Count text pixels vertically, find gaps
    """
    
    print("\nSTEP 2: Detecting Column Boundaries")
    print("=" * 50)
    
    if opencv_img is None:
        print("No image to process")
        return None, None
    
    # Convert to grayscale for easier analysis
    # WHY: Color doesn't matter for finding text vs whitespace
    # WHAT: Reduces complexity from 3 color channels to 1 brightness channel
    
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale")
    
    # Create binary image (pure black text on white background)
    # WHY: Makes it easier to count "text pixels" vs "empty pixels"
    # HOW: Pixels below threshold (dark) = text, above threshold = background
    
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    print("Created binary image (text = white, background = black)")
    
    # Calculate vertical histogram
    # WHY: Sum up all text pixels in each vertical column
    # RESULT: Array showing "how much text" exists at each horizontal position
    
    vertical_hist = np.sum(binary, axis=0)  # Sum along vertical axis
    
    print(f"Histogram calculated - {len(vertical_hist)} data points")
    print(f"Max text density: {np.max(vertical_hist)}")
    print(f"Min text density: {np.min(vertical_hist)}")
    
    # Plot the histogram to visualize column boundaries
    # WHY: Visual inspection helps us understand the pattern
    # WHAT: Shows peaks (text-heavy areas) and valleys (gaps)
    
    plt.figure(figsize=(12, 4))
    plt.plot(vertical_hist, linewidth=2, color='blue')
    plt.title('Vertical Text Density - Peaks = Text, Valleys = Gaps')
    plt.xlabel('Horizontal Position (pixels)')
    plt.ylabel('Text Density (pixel count)')
    plt.grid(True, alpha=0.3)
    
    # Add threshold line to show gap detection level
    threshold = np.max(vertical_hist) * 0.05
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Gap Threshold ({threshold:.1f})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('histogram_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Histogram plot saved as 'histogram_analysis.png'")
    
    return vertical_hist, binary

def find_column_splits(vertical_hist, min_gap_width=20):
    """
    STEP 3: Identify where columns are separated
    
    WHY: We need exact pixel coordinates to create extraction boxes
    ALGORITHM: Find continuous regions with very little text (gaps between columns)
    """
    
    print("\n STEP 3: Finding Column Split Points")
    print("=" * 50)
    
    if vertical_hist is None:
        print("No histogram data to process")
        return []
    
    # Define what counts as "low text density" (gap between columns)
    # WHY: Need threshold to distinguish between "little text" and "no text"
    # RULE: If less than 5% of maximum density = probably a gap
    
    threshold = np.max(vertical_hist) * 0.05
    print(f"Gap detection threshold: {threshold:.1f}")
    
    # Find all positions below threshold (potential gaps)
    gap_positions = np.where(vertical_hist < threshold)[0]
    
    if len(gap_positions) == 0:
        print("No clear gaps found - might be single column")
        return []
    
    print(f"Found {len(gap_positions)} potential gap pixels")
    
    # Group consecutive gap positions into gap regions
    # WHY: A column gap is usually several pixels wide, not just 1 pixel
    # HOW: Find continuous sequences of low-density pixels
    
    gaps = []
    current_gap_start = gap_positions[0]
    current_gap_end = gap_positions[0]
    
    for i in range(1, len(gap_positions)):
        if gap_positions[i] == gap_positions[i-1] + 1:
            # Continue current gap
            current_gap_end = gap_positions[i]
        else:
            # End current gap, start new one
            gap_width = current_gap_end - current_gap_start + 1
            
            if gap_width >= min_gap_width:  # Only consider wide enough gaps
                gap_center = (current_gap_start + current_gap_end) // 2
                gaps.append({
                    'start': current_gap_start,
                    'end': current_gap_end,
                    'center': gap_center,
                    'width': gap_width
                })
                print(f"Found gap: pixels {current_gap_start}-{current_gap_end} (width: {gap_width})")
            
            current_gap_start = gap_positions[i]
            current_gap_end = gap_positions[i]
    
    # Don't forget the last gap
    gap_width = current_gap_end - current_gap_start + 1
    if gap_width >= min_gap_width:
        gap_center = (current_gap_start + current_gap_end) // 2
        gaps.append({
            'start': current_gap_start,
            'end': current_gap_end,
            'center': gap_center,
            'width': gap_width
        })
        print(f"Found gap: pixels {current_gap_start}-{current_gap_end} (width: {gap_width})")
    
    print(f"Total column gaps detected: {len(gaps)}")
    
    if len(gaps) == 0:
        print("CONCLUSION: Single column document detected")
    elif len(gaps) == 1:
        print("CONCLUSION: Two column document detected")
    else:
        print(f"CONCLUSION: {len(gaps) + 1} column document detected")
    
    return gaps

def main():
    """Main function to run the complete column detection analysis"""
    
    print("CV COLUMN DETECTION ANALYSIS")
    print("=" * 60)
    
    # Get PDF path from user
    pdf_path = input("Enter path to your CV PDF file: ").strip()
    
    if not pdf_path:
        pdf_path = "F:\\Resume.pdf"  # Default path
        print(f"Using default path: {pdf_path}")
    
    # Step 1: Analyze layout
    page, opencv_img, width, height = analyze_cv_layout(pdf_path)
    
    if opencv_img is None:
        print("Failed to load PDF. Please check the file path.")
        return
    
    # Step 2: Detect column boundaries
    vertical_hist, binary = detect_column_boundaries(opencv_img)
    
    # Step 3: Find column splits
    gaps = find_column_splits(vertical_hist)
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Files created:")
    print("   • layout_analysis.png - Visual layout of your CV")
    print("   • histogram_analysis.png - Column detection plot")
    print(f"Page dimensions: {width} x {height} points")
    print(f"Detected layout: {len(gaps) + 1} columns" if gaps else "Detected layout: Single column or unclear separation")
    
    if gaps:
        print("\n Column boundaries (in pixels):")
        print("   Column 1: Start → ", gaps[0]['start'])
        for i, gap in enumerate(gaps):
            print(f"   Column {i+2}: {gap['end']} → ", gaps[i+1]['start'] if i+1 < len(gaps) else "End")
    
    print("\n NEXT STEP: Examine the generated images to verify column detection!")
    print("   1. Check layout_analysis.png - does it show your CV clearly?")
    print("   2. Check histogram_analysis.png - do the valleys align with column gaps?")

if __name__ == "__main__":
    main()
