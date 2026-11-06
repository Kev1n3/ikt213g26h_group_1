import cv2


def put_text_outlined(frame, text, org, font, scale, color, thickness, outline_color=(0, 0, 0), outline_radius=1):
    """draws text with an outline so its easier to read"""
    x, y = org
    for dx in range(-outline_radius, outline_radius + 1):
        for dy in range(-outline_radius, outline_radius + 1):
            if (dx, dy) != (0, 0) and dx * dx + dy * dy <= outline_radius * outline_radius + 1:
                cv2.putText(frame, text, (x + dx, y + dy), font, scale, outline_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def certainty_color(certainty):
    """BGR color: green > 0.2, yellow > 0.1, orange > 0.05, else red"""
    if certainty > 0.2:
        return (0, 255, 0)
    if certainty > 0.1:
        return (0, 255, 255)
    if certainty > 0.05:
        return (0, 165, 255)
    return (0, 0, 255)


def draw_facebox(frame, face_boxes, name, similarity=None, recognized=True, certainty=None):
    """Draws a colored box around the face, with the name of guessed person and other info"""

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

    # certainty text (smaller, colored) goes above the label
    cert_font_scale = 0.45
    cert_thickness = 1
    cert_label = f"certainty {certainty:.2f}" if certainty is not None else None
    cert_space = 0
    if cert_label:
        (_, cert_height), _ = cv2.getTextSize(cert_label, font, cert_font_scale, cert_thickness)
        cert_space = cert_height + 6

    # push label down if needed, so top text still gets shown if the frame is partly outside the top part of screen
    text_y = max(y0 - 10, text_height + 10 + cert_space)

    if cert_label:
        cert_pos = (x0 + 3, text_y - text_height - 8)
        cert_color = certainty_color(certainty)
        put_text_outlined(frame, cert_label, cert_pos, font, cert_font_scale, cert_color, cert_thickness)

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


def draw_tracked_faces(frame, tracked_faces, sface_threshold):
    """Draws a debug box for every tracked face"""
    for face in tracked_faces:
        name, score = face.best_match
        is_recognized = score >= sface_threshold
        if not is_recognized:
            name = "Unknown"

        draw_facebox_debug( # also shows ppl that almost matched
            frame         = frame,
            face_boxes    = face.box,
            name          = name,
            similarity    = score,
            recognized    = is_recognized,
            person_scores = face.scores,
            certainty     = face.certainty
        )
    return frame


def draw_facebox_debug(frame, face_boxes, name, similarity=None, recognized=True, person_scores=None, certainty=None):
    """Same as draw_facebox, but also lists every candidate person and score on the right side of the box"""
    draw_facebox(frame, face_boxes, name, similarity, recognized, certainty)

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

        put_text_outlined(frame, line, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return frame