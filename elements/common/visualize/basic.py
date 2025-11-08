import cv2


def draw_rect(img, y1, x1, y2, x2, text=None, color=(255, 0, 0), width=2):
    if not img.data.contiguous:
        print("Warning: the array is not contiguous, the rectangle will probably not be visible.")
    y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, width)

    if text is not None:
        text = str(text)
        text_size = cv2.getTextSize(f"{text}", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(img, (x1 + 1, y1 + 1), (x1 + text_size[0] + 4, y1 + text_size[1] + 5), 0, -1)
        cv2.putText(img, f"{text}", (x1 + 1, y1 + text_size[1] + 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)


def draw_point(img, y, x, radius: int = 10, thickness: int = 1, text=None, color=(255, 0, 0)):
    if not img.data.contiguous:
        print("Warning: the array is not contiguous, the rectangle will probably not be visible.")
    y, x = int(y), int(x)
    cv2.circle(img, (x, y), radius=radius, color=color, thickness=thickness)

    if text is not None:
        text = str(text)
        text_size = cv2.getTextSize(f"{text}", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(img, (x, y), (x + text_size[0] + 3, y + text_size[1] + 4), 0, -1)
        cv2.putText(img, f"{text}", (x, y + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
