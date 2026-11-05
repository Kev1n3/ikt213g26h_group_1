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


def draw_facebox_debug(frame, face_boxes, name, similarity=None, recognized=True, person_scores=None):
    """Same as draw_facebox, but also lists every candidate person and score on the right side of the box"""
    draw_facebox(frame, face_boxes, name, similarity, recognized)

    if not person_scores:
        return frame

    x, y, w, h = map(int, face_boxes[:4])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    line_height = 16

    text_x = x + w + 6
    for i, (person_name, score) in enumerate(person_scores.items()):
        text_y = y + 12 + i * line_height
        line = f"{person_name}: {score:.2f}"

        # Dark outline for readability
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return frame