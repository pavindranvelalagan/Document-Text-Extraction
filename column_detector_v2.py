# column_detector.py - Refined Column Detection with Robust Post-processing
import pdfplumber
import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_cv_layout(pdf_path, dpi=200):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0] #first page of the pdf only
        img = page.to_image(resolution=dpi)
        img.save("layout_analysis.png")
        pil_img = img.original
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return page, cv_img

def compute_vertical_hist(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Smooth to suppress spikes from icons and thin lines
    # Box filter on the binary (sum axis afterward)
    smoothed = cv2.blur(binary, ksize=(9, 1))
    hist = np.sum(smoothed, axis=0).astype(np.float32)
    # Normalize for plotting/comparisons (not used for thresholding)
    return hist, binary

def find_gaps(hist, img_width,
              gap_threshold_ratio=0.05,
              min_gap_width_ratio=0.05,
              margin_ignore_ratio=0.04,
              merge_distance_ratio=0.02):
    """
    - gap_threshold_ratio: valley threshold = ratio * max(hist)
    - min_gap_width_ratio: minimum gap pixel width to consider a true separator
    - margin_ignore_ratio: ignore gaps within this ratio near left/right margins
    - merge_distance_ratio: merge gaps whose centers are closer than this ratio
    """
    max_h = np.max(hist)
    thresh = max_h * gap_threshold_ratio
    W = img_width
    min_gap_width = int(W * min_gap_width_ratio)
    margin_px = int(W * margin_ignore_ratio)
    merge_px = int(W * merge_distance_ratio)

    below = np.where(hist < thresh)[0]
    gaps = []
    if below.size == 0:
        return gaps, thresh

    start = below[0]
    prev = below[0]
    for x in below[1:]:
        if x == prev + 1:
            prev = x
        else:
            width = prev - start + 1
            if width >= min_gap_width:
                # ignore margins
                if prev < margin_px or start > (W - margin_px):
                    pass
                else:
                    gaps.append({'start': int(start), 'end': int(prev),
                                 'center': int((start + prev) // 2),
                                 'width': int(width)})
            start = x
            prev = x
    # last gap
    width = prev - start + 1
    if width >= min_gap_width:
        if prev >= margin_px and start <= (W - margin_px):
            gaps.append({'start': int(start), 'end': int(prev),
                         'center': int((start + prev) // 2),
                         'width': int(width)})

    # merge nearby gaps (handles thin visual separators near the true split)
    if not gaps:
        return gaps, thresh
    gaps.sort(key=lambda g: g['center'])
    merged = [gaps]
    for g in gaps[1:]:
        if g['center'] - merged[-1]['center'] <= merge_px:
            merged[-1]['start'] = min(merged[-1]['start'], g['start'])
            merged[-1]['end'] = max(merged[-1]['end'], g['end'])
            merged[-1]['width'] = merged[-1]['end'] - merged[-1]['start'] + 1
            merged[-1]['center'] = (merged[-1]['start'] + merged[-1]['end']) // 2
        else:
            merged.append(g)
    return merged, thresh

def choose_splits_two_columns(hist, gaps, img_width,
                              min_band_ratio=0.10):
    """
    Choose the single best split for two columns:
    - For each candidate gap, measure text mass on left/right sides
    - Keep the gap that yields two substantial text bands
    """
    W = img_width
    min_band_width = int(W * min_band_ratio)
    if not gaps:
        return []

    best = None
    best_score = -1.0
    cum = np.cumsum(hist)  # prefix sums for fast band mass

    for g in gaps:
        split = g['center']
        left_width = split
        right_width = W - split
        if left_width < min_band_width or right_width < min_band_width:
            continue
        left_mass = cum[split-1] if split > 0 else 0.0
        right_mass = cum[-1] - cum[split-1] if split > 0 else cum[-1]
        # score: prefer balanced and high mass bands
        total = left_mass + right_mass + 1e-6
        balance = 1.0 - abs(left_mass - right_mass) / total
        score = balance * total
        if score > best_score:
            best_score = score
            best = g

    return [best] if best else []

def plot_hist(hist, thresh, splits):
    plt.figure(figsize=(12, 4))
    plt.plot(hist, linewidth=2, color='blue')
    plt.axhline(y=thresh, color='red', linestyle='--', alpha=0.7, label=f'Threshold {thresh:.1f}')
    for s in splits:
        plt.axvline(x=s['center'], color='green', linestyle='-', alpha=0.8, label='Chosen split')
    plt.title('Vertical Text Density with Chosen Split(s)')
    plt.xlabel('Horizontal Position (pixels)')
    plt.ylabel('Text Density (pixel count)')
    plt.grid(True, alpha=0.3)
    # deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys())
    plt.tight_layout()
    plt.savefig('histogram_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    pdf_path = input("Enter path to your CV PDF file: ").strip()
    page, cv_img = analyze_cv_layout(pdf_path, dpi=200)
    H, W = cv_img.shape[:2]
    print(f"Image size: {W}x{H}")

    hist, binary = compute_vertical_hist(cv_img)
    gaps, thresh = find_gaps(
        hist, W,
        gap_threshold_ratio=0.05,      # valley threshold at 5% of max
        min_gap_width_ratio=0.05,      # gaps must be ≥5% of width
        margin_ignore_ratio=0.04,      # ignore 4% margins
        merge_distance_ratio=0.02      # merge gaps closer than 2% width
    )
    print(f"Candidate gaps after filtering/merging: {len(gaps)}")
    for g in gaps:
        print(f"  gap {g['start']}-{g['end']} (w={g['width']}) center={g['center']}")

    # Select one split for two columns
    chosen = choose_splits_two_columns(hist, gaps, W, min_band_ratio=0.10)
    if not chosen:
        print("No robust two-column split found. Treat as single column or adjust ratios.")
        chosen = []

    # Draw result
    plot_hist(hist, thresh, chosen)

    # Report column boxes for extraction (left then right)
    if chosen:
        c = chosen[0]['center'] #chosen is a list with one dict
        # convert pixel x to pdf coordinate x using ratio between image and pdf page width
        px_per_pt = W / page.width
        left_bbox_px = (0, 0, c, H)
        right_bbox_px = (c, 0, W, H)
        print("\nColumn pixel boxes (for debug):")
        print("  Left :", left_bbox_px)
        print("  Right:", right_bbox_px)

        # Map pixel x to PDF points for pdfplumber crop
        left_bbox_pts = (0, 0, c / px_per_pt, page.height)
        right_bbox_pts = (c / px_per_pt, 0, page.width, page.height)
        print("\nColumn PDF-point boxes (use with pdfplumber.crop):")
        print("  Left :", tuple(round(v, 2) for v in left_bbox_pts))
        print("  Right:", tuple(round(v, 2) for v in right_bbox_pts))

        # Optional: immediate extraction preview
        with pdfplumber.open(pdf_path) as pdf:
            p = pdf.pages[0] #open first page only
            left_text = p.crop(left_bbox_pts).extract_text() or ""
            right_text = p.crop(right_bbox_pts).extract_text() or ""
        with open("extracted_left.txt", "w", encoding="utf-8") as f:
            f.write(left_text)
        with open("extracted_right.txt", "w", encoding="utf-8") as f:
            f.write(right_text)
        print("\nSaved extracted_left.txt and extracted_right.txt")

if __name__ == "__main__":
    main()
