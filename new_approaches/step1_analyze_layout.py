# step1_analyze_layout.py
import fitz
import json

def analyze_pdf_layout(pdf_path):
    """Analyze the layout structure of a PDF to understand column positioning"""
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    # Get page dimensions
    page_width = page.rect.width
    page_height = page.rect.height
    
    print(f"Page dimensions: {page_width:.1f} x {page_height:.1f}")
    
    # Extract text blocks with coordinates
    blocks = page.get_text("blocks")
    print(f"Found {len(blocks)} text blocks")
    
    # Analyze block positions
    block_info = []
    for i, block in enumerate(blocks):
        x0, y0, x1, y1, text, block_type, block_no = block
        
        block_data = {
            'block_id': i,
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'width': x1 - x0,
            'height': y1 - y0,
            'text_preview': text[:100].replace('\n', ' ').strip()
        }
        block_info.append(block_data)
        
        print(f"Block {i}: ({x0:.1f},{y0:.1f}) -> ({x1:.1f},{y1:.1f})")
        print(f"  Text: {text[:80].replace('\n', ' ')}...")
        print()
    
    doc.close()
    return block_info, page_width, page_height

# Test with your CV
if __name__ == "__main__":
    blocks, width, height = analyze_pdf_layout("F:/Cogntix/Unblit/AI/SampleCVs/Sample1.pdf")
