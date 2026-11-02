import cv2
def draw_facebox(frame, face_boxes, name, similarity=None, recognized=True):
    x0, y0, x1, y1 = map(int, face_boxes[:4])

    color = (0, 255, 0) if recognized else (0, 0, 255)
    thickness = 2

    # Draws the rectangle around the face
    cv2.rectangle(frame, (x0, y0), (x0 + x1, y0 + y1), color, thickness)

    label = name
    if similarity is not None:
        label += f" ({similarity:.2f})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    text_y = max(y0 - 10, text_height + 10)

    cv2.rectangle(
        frame,
        (x0, text_y - text_height - 4),
        (x0 + text_width + 6, text_y + baseline),
        color,
        -1
    )
    cv2.putText(
        frame,
        label,
        (x0 + 3, text_y - 2),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )
    return frame