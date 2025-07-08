import cv2
import numpy as np
import time
import mido
import pretty_midi
import math
from sklearn.cluster import KMeans
import pygame
import os

# --- CONFIGURATION ---
IMAGE_PATH = "image.png"
MIDI_PATH = "Jazz_Suite_No._2_II._Lyric_Waltz__Dmitri_Shostakovich.mid"
AUDIO_PATH = "Jazz_Suite_No._2_II._Lyric_Waltz__Dmitri_Shostakovich.mp3"
FRAME_FOLDER = "frames"
OUTPUT_VIDEO = "output_video.mp4"
FPS = 30
NUM_COLOR_GROUPS = 11
REVEAL_STEP = 10
FADE_DURATION = 1.0
GLOW_RADIUS = 10
GLOW_COLOR = (0, 255, 200)
GLOW_MAX_ALPHA = 0.2
PULSE_SPEED = 15.0
PULSE_AMPLITUDE = 0.1
SCALE_IMAGE = 0.08

# --- UTILITY FUNCTIONS ---

def is_dark_pixel(pixel, threshold=30):
    luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
    return luminance < threshold

def group_pixels_by_color(image, num_groups=8):
    h, w, _ = image.shape
    flat_img = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_groups, random_state=42)
    labels = kmeans.fit_predict(flat_img)
    pixel_groups = [[] for _ in range(num_groups)]
    for idx, label in enumerate(labels):
        y, x = divmod(idx, w)
        pixel_groups[label].append((y, x))
    return pixel_groups

def make_glow_circle(radius, max_alpha=0.6, color=(255, 255, 200)):
    size = radius * 2 + 1
    glow = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            dy, dx = y - radius, x - radius
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= radius:
                alpha = max_alpha * (1 - dist / radius) ** 2
                glow[y, x, :3] = color
                glow[y, x, 3] = int(alpha * 255)
    return glow

def overlay_rgba(src, overlay, pos):
    x, y = pos
    h, w = overlay.shape[:2]
    if y + h > src.shape[0]:
        h = src.shape[0] - y
    if x + w > src.shape[1]:
        w = src.shape[1] - x
    if h <= 0 or w <= 0 or overlay.shape[0] == 0 or overlay.shape[1] == 0:
        return
    overlay_cropped = overlay[:h, :w]
    roi = src[y:y+h, x:x+w]
    if overlay_cropped.shape[0] != roi.shape[0] or overlay_cropped.shape[1] != roi.shape[1]:
        min_h = min(overlay_cropped.shape[0], roi.shape[0])
        min_w = min(overlay_cropped.shape[1], roi.shape[1])
        overlay_cropped = overlay_cropped[:min_h, :min_w]
        roi = roi[:min_h, :min_w]
    alpha = overlay_cropped[..., 3] / 255.0
    alpha_exp = alpha[..., np.newaxis]
    roi[...] = (alpha_exp * overlay_cropped[..., :3] + (1 - alpha_exp) * roi).astype(np.uint8)
    src[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

# --- INITIALIZATION ---

def load_and_prepare_images():
    image2 = cv2.imread(IMAGE_PATH)
    height_first, width_first, _ = image2.shape
    image = cv2.resize(image2, (0, 0), fx=SCALE_IMAGE, fy=SCALE_IMAGE, interpolation=cv2.INTER_AREA)
    black_image = np.zeros_like(image)
    return image, black_image, image2, height_first, width_first

def prepare_audio_and_midi():
    midi = pretty_midi.PrettyMIDI(MIDI_PATH)
    midi_file = mido.MidiFile(MIDI_PATH)
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_PATH)
    return midi, midi_file

def setup_frame_folder():
    os.makedirs(FRAME_FOLDER, exist_ok=True)

# --- MAIN REVEAL LOOP ---

def reveal_image_with_music():
    image, black_image, image2, height_first, width_first = load_and_prepare_images()
    pixel_groups = group_pixels_by_color(image, num_groups=NUM_COLOR_GROUPS)
    group_index = 0
    group_cursor = 0
    pulsing_pixels = {}
    fade_circles = {}
    frame_idx = 0

    dummy_image = cv2.resize(black_image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Revealing Image", dummy_image)
    time.sleep(1)

    midi, midi_file = prepare_audio_and_midi()
    pygame.mixer.music.play()
    start_time = time.time()

    scale_x = width_first / image.shape[1]
    scale_y = height_first / image.shape[0]
    glow_circle = make_glow_circle(GLOW_RADIUS, max_alpha=GLOW_MAX_ALPHA, color=GLOW_COLOR)

    for msg in midi_file.play():
        current_time = time.time() - start_time

        # Fade out glow circles
        to_remove = []
        for key in fade_circles:
            fade_circles[key] += 0.01
            if fade_circles[key] > FADE_DURATION:
                to_remove.append(key)
        for key in to_remove:
            fade_circles.pop(key)

        new_pixels_this_frame = []

        # Reveal logic
        if msg.type == 'note_on' and msg.velocity > 0:
            revealed = 0
            # Count revealed pixels
            total_pixels = image.shape[0] * image.shape[1]
            revealed_pixels = np.count_nonzero(np.any(black_image != 0, axis=2))
            revealed_ratio = revealed_pixels / total_pixels

            if revealed_ratio <= 0.5 and msg.velocity >= 95:
                # Use a set to track unrevealed, non-dark pixels for efficient random selection
                if not hasattr(reveal_image_with_music, "unrevealed_pixels"):
                    reveal_image_with_music.unrevealed_pixels = {
                    (y, x)
                    for y in range(image.shape[0])
                    for x in range(image.shape[1])
                    if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
                    }
                unrevealed = reveal_image_with_music.unrevealed_pixels
                if unrevealed:
                    choices = list(unrevealed)
                    np.random.shuffle(choices)
                    for y, x in choices:
                        black_image[y, x] = image[y, x]
                        pulsing_pixels[(y, x)] = 0.0
                        new_pixels_this_frame.append((y, x))
                        fade_circles[(y, x)] = 0.0
                        unrevealed.remove((y, x))
                        revealed += 1
                        if revealed >= REVEAL_STEP:
                            break
            else:
                if group_index < len(pixel_groups):
                    group = pixel_groups[group_index]
                    while group_cursor < len(group) and revealed < REVEAL_STEP:
                        y, x = group[group_cursor]
                        if not is_dark_pixel(image[y, x]) and np.all(black_image[y, x] == 0):
                            black_image[y, x] = image[y, x]
                            pulsing_pixels[(y, x)] = 0.0
                            new_pixels_this_frame.append((y, x))
                            fade_circles[(y, x)] = 0.0
                            revealed += 1
                        group_cursor += 1
                    if group_cursor >= len(group):
                        group_index += 1
                        group_cursor = 0

        # Update pulsing pixels
        for (y, x), phase in list(pulsing_pixels.items()):
            phase += 2 * math.pi * PULSE_SPEED * 0.01
            pulse = (math.sin(phase) * 0.5 + 0.5) * PULSE_AMPLITUDE + (1 - PULSE_AMPLITUDE)
            orig_color = image[y, x].astype(np.float32)
            pulsed_color = np.clip(orig_color * pulse, 0, 255)
            black_image[y, x] = pulsed_color.astype(np.uint8)
            pulsing_pixels[(y, x)] = phase

        # Prepare frame for display and saving
        big_version = cv2.resize(black_image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)

        # Overlay glow circles
        for (y, x), age in fade_circles.items():
            alpha_factor = max(0, 1 - age / FADE_DURATION)
            center_x = int(round(x * scale_x + scale_x / 2)) - GLOW_RADIUS
            center_y = int(round(y * scale_y + scale_y / 2)) - GLOW_RADIUS
            glow_with_fade = glow_circle.copy()
            glow_with_fade[..., 3] = (glow_with_fade[..., 3].astype(np.float32) * alpha_factor).astype(np.uint8)
            overlay_rgba(big_version, glow_with_fade, (center_x, center_y))

        cv2.imshow("Revealing Image", big_version)
        frame_path = os.path.join(FRAME_FOLDER, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_path, big_version)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- MAIN ENTRY POINT ---

if __name__ == "__main__":
    setup_frame_folder()
    reveal_image_with_music()
