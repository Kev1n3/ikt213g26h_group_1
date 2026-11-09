import cv2

from src.TrackedFace import TrackedFace


def put_text_outlined(frame, text, org, font, scale, color, thickness, outline_color=(0, 0, 0), outline_radius=1):
    """draws text with an outline so its easier to read"""
    x, y = org
    for dx in range(-outline_radius, outline_radius + 1):
        for dy in range(-outline_radius, outline_radius + 1):
            if (dx, dy) != (0, 0) and dx * dx + dy * dy <= outline_radius * outline_radius + 1:
                cv2.putText(frame, text, (x + dx, y + dy), font, scale, outline_color, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


# color limits for confidence and similarity. orange -> yellow is at sface_threshold
UNKNOWN_BELOW = 0.25 # red, shown as "Unknown"
SURE_FROM = 0.5 # green, shown without "?"
SURE_CERTAINTY = 0.025 # green also needs a clear winner. below this it cant be higher than yellow

def score_color(value, limits=(0.25, 0.5, 0.75)):
    """BGR color for a value. limits = (orange from, yellow from, green from), else red"""
    orange_from, yellow_from, green_from = limits
    if value >= green_from:
        return (0, 255, 0)
    if value >= yellow_from:
        return (0, 255, 255)
    if value >= orange_from:
        return (0, 165, 255)
    return (0, 0, 255)


def label_text_color(background):
    """black text on bright backgrounds (yellow), white otherwise"""
    b, g, r = background
    brightness = 0.114 * b + 0.587 * g + 0.299 * r # perceived brightness, eyes are most sensitive to green
    return (0, 0, 0) if brightness > 200 else (255, 255, 255)


SPOOF_COLOR = (140, 0, 210) # BGR, red-purple


def draw_dashed_rect(frame, x, y, w, h, color, thickness=2, dash=8):
    """draws a dashed rectangle. dash = length of each line piece, same length gap between"""
    # top and bottom
    for i in range(x, x + w, dash * 2):
        end = min(i + dash, x + w) # last piece cant go past the corner
        cv2.line(frame, (i, y), (end, y), color, thickness)
        cv2.line(frame, (i, y + h), (end, y + h), color, thickness)
    # left and right
    for j in range(y, y + h, dash * 2):
        end = min(j + dash, y + h)
        cv2.line(frame, (x, j), (x, end), color, thickness)
        cv2.line(frame, (x + w, j), (x + w, end), color, thickness)


def draw_facebox(frame, face_boxes, label, color, stats=None, thickness=2, spoofed=False):
    """Draws a colored box around the face, with the name of guessed person and other small colored stats above it.
    stats: list<(text, color)>. thickness: line width of the box"""

    x0, y0, x1, y1 = map(int, face_boxes[:4])

    # Draws the rectangle around the face
    cv2.rectangle(frame, (x0, y0), (x0 + x1, y0 + y1), color, thickness)
    if spoofed:
        draw_dashed_rect(frame, x0, y0, x1, y1, (0, 0, 255), thickness) # red dashes

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # stats text (smaller, colored) goes above the label
    stats_font_scale = 0.45
    stats_thickness = 1
    stats_gap = 10 # px between each stat
    stats_space = 0
    if stats:
        (_, stats_height), _ = cv2.getTextSize(stats[0][0], font, stats_font_scale, stats_thickness)
        stats_space = stats_height + 6

    # push label down if needed, so top text still gets shown if the frame is partly outside the top part of screen
    text_y = max(y0 - 10, text_height + 10 + stats_space)

    if stats:
        stat_x = x0 + 3
        stat_y = text_y - text_height - 8
        for stat_text, stat_color in stats:
            put_text_outlined(frame, stat_text, (stat_x, stat_y), font, stats_font_scale, stat_color, stats_thickness)
            (stat_width, _), _ = cv2.getTextSize(stat_text, font, stats_font_scale, stats_thickness)
            stat_x += stat_width + stats_gap

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
        label_text_color(color),
        font_thickness,
        cv2.LINE_AA
    )

    # tag right of the name label, own color so it doesnt mix with the confidence colors
    if spoofed:
        tag = "SPOOFED"
        (tag_width, _), _ = cv2.getTextSize(tag, font, font_scale, font_thickness)
        tag_x = x0 + text_width + 6 + 4 # small gap
        cv2.rectangle(frame, (tag_x, text_y - text_height - 4), (tag_x + tag_width + 6, text_y + baseline), SPOOF_COLOR, -1)
        cv2.putText(frame, tag, (tag_x + 3, text_y - 2), font, font_scale, label_text_color(SPOOF_COLOR), font_thickness, cv2.LINE_AA)
    return frame


def draw_tracked_faces(frame, tracked_faces: list[TrackedFace]):
    """Draws a debug box for every tracked face. box + label colored by confidence, small stats colored by their own factor"""
    for face in tracked_faces:
        name, similarity = face.best_match
        confidence = face.confidence # if 2 persons have almost the same score, confidence will be low
        limits = (UNKNOWN_BELOW, face.sface_threshold, SURE_FROM)
        color = score_color(confidence, limits)
        is_sure = confidence >= SURE_FROM and face.certainty >= SURE_CERTAINTY
        if confidence >= SURE_FROM and not is_sure:
            color = (0, 255, 255) # yellow, high confidence but no clear winner

        # red -> Unknown, yellow/orange -> "Name?", green -> "Name"
        if confidence < UNKNOWN_BELOW:
            name = "Unknown"
        elif not is_sure:
            name += "?"

        stats = [
            (f"similar {similarity:.2f}", score_color(similarity, limits)),
            (f"certain {face.certainty:.2f}", score_color(face.certainty_strength)),
            (f"{face.spoofing["label"]}: {face.spoofing["real_score"]:.2f}", score_color(face.spoofing["real_score"])),
        ]
        if face.is_missing:
            stats.append((f"last seen {face.seconds_missing:.1f}s", score_color(face.time_factor)))

        draw_facebox_debug( # also shows ppl that almost matched
            frame         = frame,
            face_boxes    = face.box,
            label         = f"{name} ({confidence:.2f})",
            color         = color,
            stats         = stats,
            person_scores = face.scores,
            thickness     = 1 if face.is_missing else 2, # thinner box when not detected this frame
            spoofed       = face.spoofing["label"] == "Fake" and face.spoofed_frames >= 3
        )
    return frame


def draw_facebox_debug(frame, face_boxes, label, color, stats=None, person_scores=None, thickness=2, spoofed=False):
    """Same as draw_facebox, but also lists every candidate person and score on the right side of the box"""
    draw_facebox(frame, face_boxes, label, color, stats, thickness, spoofed=spoofed)

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