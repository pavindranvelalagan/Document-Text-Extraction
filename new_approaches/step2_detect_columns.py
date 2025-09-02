# step2_detect_columns.py
import fitz
import numpy as np

class ColumnDetector:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.page = self.doc[0]
        self.page_width = self.page.rect.width
        self.blocks = self.page.get_text("blocks")
    
    def detect_columns(self):
        """Automatically detect column boundaries"""
        # Get all x-coordinates where text blocks start
        x_starts = [block[0] for block in self.blocks if len(block[4].strip()) > 10]
        
        if len(x_starts) < 2:
            return [self.page_width / 2]  # Default: single column split
        
        # Use clustering to find column boundaries
        x_starts.sort()
        
        # Find the largest gap in x-coordinates (likely column separator)
        gaps = []
        for i in range(len(x_starts) - 1):
            gap = x_starts[i + 1] - x_starts[i]
            if gap > 30:  # Minimum gap threshold
                gaps.append((gap, (x_starts[i] + x_starts[i + 1]) / 2))
        
        if not gaps:
            return [self.page_width / 2]
        
        # Return the position of the largest gap
        largest_gap = max(gaps, key=lambda x: x[0])
        column_boundary = largest_gap[1]
        
        print(f"Detected column boundary at x = {column_boundary:.1f}")
        return [column_boundary]
    
    def separate_blocks_by_columns(self, column_boundaries):
        """Separate text blocks into columns based on boundaries"""
        columns = [[] for _ in range(len(column_boundaries) + 1)]
        
        for block in self.blocks:
            x0, y0, x1, y1, text, block_type, block_no = block
            
            if len(text.strip()) < 5:  # Skip empty blocks
                continue
            
            # Determine which column this block belongs to
            block_center_x = (x0 + x1) / 2
            column_index = 0
            
            for i, boundary in enumerate(column_boundaries):
                if block_center_x > boundary:
                    column_index = i + 1
                else:
                    break
            
            columns[column_index].append({
                'coordinates': (x0, y0, x1, y1),
                'text': text,
                'y_position': y0  # For sorting by vertical position
            })
        
        # Sort each column by vertical position (top to bottom)
        for column in columns:
            column.sort(key=lambda x: x['y_position'])
        
        return columns
    
    def close(self):
        self.doc.close()

# Test column detection
if __name__ == "__main__":
    detector = ColumnDetector("F:/Cogntix/Unblit/AI/SampleCVs/Sample1.pdf")
    boundaries = detector.detect_columns()
    columns = detector.separate_blocks_by_columns(boundaries)
    
    for i, column in enumerate(columns):
        print(f"\n=== COLUMN {i + 1} ===")
        for block in column:
            print(f"Text: {block['text'][:100].replace('\n', ' ')}...")
    
    detector.close()
