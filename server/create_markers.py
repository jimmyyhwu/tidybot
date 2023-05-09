import tempfile
from pathlib import Path
import cv2 as cv
from fpdf import FPDF
from PIL import Image
import utils
from constants import MARKER_PARAMS, MARKER_DICT_ID, MARKER_IDS

def create_markers():
    # Paper params
    output_dir = 'printouts'
    pdf_name = 'markers.pdf'
    orientation = 'P'
    paper_params = utils.get_paper_params(orientation)

    # Marker params
    marker_length_pixels = 6
    marker_length_mm = 1000 * MARKER_PARAMS['marker_length']
    sticker_length_mm = 1000 * MARKER_PARAMS['sticker_length']
    sticker_spacing_mm = 10
    scale_factor = marker_length_mm / (paper_params['mm_per_printed_pixel'] * marker_length_pixels)
    stickers_per_row = int((paper_params['width_mm'] - 2 * paper_params['margin_mm'] + sticker_spacing_mm) / (sticker_length_mm + sticker_spacing_mm))
    rows_per_page = int((paper_params['height_mm'] - 2 * paper_params['margin_mm'] + sticker_spacing_mm) / (sticker_length_mm + sticker_spacing_mm))
    aruco_dict = cv.aruco.Dictionary_get(MARKER_DICT_ID)

    # Create PDF
    pdf = FPDF(orientation, 'mm', 'letter')
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        for marker_idx, marker_id in enumerate(MARKER_IDS):
            i = marker_idx % (stickers_per_row * rows_per_page)
            if i == 0:
                pdf.add_page()
            image_path = str(Path(tmp_dir_name) / f'{marker_id}.png')
            Image.fromarray(cv.aruco.drawMarker(aruco_dict, marker_id, int(scale_factor * marker_length_pixels))).save(image_path)
            center_x = paper_params['margin_mm'] + sticker_length_mm / 2 + (sticker_length_mm + sticker_spacing_mm) * (i % stickers_per_row)
            center_y = paper_params['margin_mm'] + sticker_length_mm / 2 + (sticker_length_mm + sticker_spacing_mm) * (i // stickers_per_row)
            pdf.rect(
                x=(center_x - sticker_length_mm / 2 - pdf.line_width / 2),
                y=(center_y - sticker_length_mm / 2 - pdf.line_width / 2),
                w=(sticker_length_mm + pdf.line_width),
                h=(sticker_length_mm + pdf.line_width)
            )
            pdf.image(image_path, x=(center_x - marker_length_mm / 2), y=(center_y - marker_length_mm / 2), w=marker_length_mm, h=marker_length_mm)

    # Save PDF
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    pdf.output(output_dir / pdf_name)

if __name__ == '__main__':
    create_markers()
