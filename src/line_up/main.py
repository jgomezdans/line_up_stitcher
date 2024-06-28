import datetime as dt
import sys
from pathlib import Path

import cv2

from .line_up import ImageStitcher

def print_help():
    print("Usage: line_up [FILES...]")
    print("Stitches up a bunch of overlapping image files.")
    print("\nOptions:")
    print("  -h, --help  Show this help message and exit")

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print_help()
        return
    
    files = sys.argv[1:]
    
    all_files_ok = True
    for file in files:
        path = Path(file)
        if not path.exists():
            print(f"Error: The file '{file}' does not exist.")
            all_files_ok = False

    if all_files_ok:
        stitcher = ImageStitcher(files, normalize=False)
        result_image = stitcher.process_images()
        timer =  f"{dt.datetime.now()}".replace(" ", "T")
        cv2.imwrite(f"superposed_image_{timer}.png", result_image)
        stitcher.save_homographies(f'homographies_{timer}.npy')
        print(f"Saved stitched up file as superposed_image_{timer}.png")
        print(f"Saved homographies as homographies_{timer}.npy")
        
if __name__ == "__main__":
    main()
