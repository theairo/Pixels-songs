import cv2
import numpy as np
import time
import mido
import pretty_midi
import math
from sklearn.cluster import KMeans
import pygame
import os
import random
import subprocess

# --- CONFIGURATION ---
IMAGE_PATH = "image.png"
MIDI_PATH = "Etude_Op.25_No.11_in_A_minor_Winter_Wind_-_F._Chopin.mid"
AUDIO_PATH = "Etude_Op.25_No.11_in_A_minor_Winter_Wind_-_F._Chopin.mp3"
FRAME_FOLDER = "frames"
OUTPUT_VIDEO = "output_video.mp4"
FPS = 30
NUM_COLOR_GROUPS = 18
REVEAL_STEP = 5
FADE_DURATION = 1.0
GLOW_RADIUS = 10
GLOW_COLOR = (0, 255, 200)
GLOW_MAX_ALPHA = 0.2
PULSE_SPEED = 15.0
PULSE_AMPLITUDE = 0.1
ON_HIGH = False
SCALE_IMAGE = 0.2


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
    # Each overlay is a dict: {'start_time': float, 'duration': float, 'opacity': float}
    overlays = []
    glow_active = False
    glow_start_time = 0

    image, black_image, image2, height_first, width_first = load_and_prepare_images()
    pixel_groups = group_pixels_by_color(image, num_groups=NUM_COLOR_GROUPS)
    group_index = 0
    group_cursor = 0
    pulsing_pixels = {}
    fade_circles = {}
    frame_idx = 0

    dummy_image = cv2.resize(black_image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Revealing Image", dummy_image)
    #time.sleep(1)

    midi, midi_file = prepare_audio_and_midi()
    pygame.mixer.music.play()
    start_time = time.time()

    scale_x = width_first / image.shape[1]
    scale_y = height_first / image.shape[0]
    glow_circle = make_glow_circle(GLOW_RADIUS, max_alpha=GLOW_MAX_ALPHA, color=GLOW_COLOR)

    # Set up video writer (temporary video without audio)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = "temp_video_no_audio.mp4"
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, FPS, (width_first, height_first))

    # Convert to absolute MIDI timing
    midi_events = []
    current_time_sec = 0
    for msg in midi_file:
        current_time_sec += msg.time
        midi_events.append((current_time_sec, msg))

    event_idx = 0
    total_events = len(midi_events)

    frame_idx = 0
    fps = FPS
    frame_duration = 1.0 / fps

    quit_flag = False

    while (event_idx < total_events or overlays or glow_active) and not quit_flag:
        now = time.time()
        elapsed = now - start_time

        # Process all MIDI messages up to current time
        while event_idx < total_events and midi_events[event_idx][0] <= elapsed:
            msg_time, msg = midi_events[event_idx]
            event_idx += 1

            new_pixels_this_frame = []

            if msg.type == 'note_on' and msg.velocity > 0:

                # Only add overlay for ~40% of notes, and only if less than 17 seconds into the song
                if random.random() < 0.4 and elapsed < 17:
                    overlays.append({
                        'start_time': time.time(),
                        'duration': 0.5,  # seconds
                        'opacity': 0.01   # 1% transparency
                    })

                revealed = 0
                total_pixels = image.shape[0] * image.shape[1]
                revealed_pixels = np.count_nonzero(np.any(black_image != 0, axis=2))
                revealed_ratio = revealed_pixels / total_pixels

                if revealed_ratio <= 0.5 and msg.velocity >= 95 and msg.note < 60:
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
                    # High note: reveal from center outward
                    if msg.note > 72 and ON_HIGH:
                        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
                        unrevealed = [
                            (y, x)
                            for y in range(image.shape[0])
                            for x in range(image.shape[1])
                            if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
                        ]
                        unrevealed.sort(key=lambda p: (p[0] - center_y) ** 2 + (p[1] - center_x) ** 2)
                        for y, x in unrevealed[:REVEAL_STEP]:
                            black_image[y, x] = image[y, x]
                            pulsing_pixels[(y, x)] = 0.0
                            new_pixels_this_frame.append((y, x))
                            fade_circles[(y, x)] = 0.0
                            revealed += 1
                    # Low note: reveal from edges inward
                    elif msg.note < 48 and ON_HIGH:
                        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
                        unrevealed = [
                            (y, x)
                            for y in range(image.shape[0])
                            for x in range(image.shape[1])
                            if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
                        ]
                        unrevealed.sort(key=lambda p: -((p[0] - center_y) ** 2 + (p[1] - center_x) ** 2))
                        for y, x in unrevealed[:REVEAL_STEP]:
                            black_image[y, x] = image[y, x]
                            pulsing_pixels[(y, x)] = 0.0
                            new_pixels_this_frame.append((y, x))
                            fade_circles[(y, x)] = 0.0
                            revealed += 1
                    # Otherwise: group-based reveal
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
        to_remove = []
        for key in fade_circles:
            fade_circles[key] += frame_duration
            if fade_circles[key] > FADE_DURATION:
                to_remove.append(key)
        for key in to_remove:
            fade_circles.pop(key)

        for (y, x), age in fade_circles.items():
            alpha_factor = max(0, 1 - age / FADE_DURATION)
            center_x = int(round(x * scale_x + scale_x / 2)) - GLOW_RADIUS
            center_y = int(round(y * scale_y + scale_y / 2)) - GLOW_RADIUS
            glow_with_fade = glow_circle.copy()
            glow_with_fade[..., 3] = (glow_with_fade[..., 3].astype(np.float32) * alpha_factor).astype(np.uint8)
            overlay_rgba(big_version, glow_with_fade, (center_x, center_y))

        # Transparent full-image overlay effects (multiple, async)
        overlay_img = cv2.resize(image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)
        overlays_to_remove = []
        for overlay in overlays:
            elapsed_overlay = time.time() - overlay['start_time']
            if elapsed_overlay <= overlay['duration']:
                fade = max(0, 1 - elapsed_overlay / overlay['duration'])
                opacity = overlay['opacity'] * fade
                temp = cv2.addWeighted(
                    big_version,
                    1.0,
                    overlay_img,
                    opacity,
                    0
                )
                # Optional: white flash for each overlay
                flash_opacity = 0.15 * fade
                white_flash = np.full_like(big_version, 255)
                temp = cv2.addWeighted(temp, 1.0, white_flash, flash_opacity, 0)
                big_version = temp
            else:
                overlays_to_remove.append(overlay)
        for overlay in overlays_to_remove:
            overlays.remove(overlay)

        cv2.imshow("Revealing Image", big_version)
        cv2.imwrite(f"{FRAME_FOLDER}/frame_{frame_idx:05d}.png", big_version)
        # Write frame to video
        video_writer.write(big_version)

        frame_idx += 1

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            quit_flag = True
            break

        time.sleep(frame_duration)
    cv2.destroyAllWindows()
    video_writer.release()

    # --- Combine video and audio using ffmpeg ---
    # This requires ffmpeg to be installed and available in PATH.
    # The final video will be OUTPUT_VIDEO with audio from AUDIO_PATH.
    cmd = [
        "ffmpeg",
        "-y",
        "-i", temp_video_path,
        "-i", AUDIO_PATH,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        OUTPUT_VIDEO
    ]
    print("Combining video and audio with ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"Video with audio saved to {OUTPUT_VIDEO}")
    # Optionally remove temp video
    os.remove(temp_video_path)

# def reveal_image_with_music():

#     overlay_active = False
#     overlay_start_time = 0
#     overlay_duration = 0.5  # seconds
#     overlay_opacity = 0.01   # 10% transparency


#     image, black_image, image2, height_first, width_first = load_and_prepare_images()
#     pixel_groups = group_pixels_by_color(image, num_groups=NUM_COLOR_GROUPS)
#     group_index = 0
#     group_cursor = 0
#     pulsing_pixels = {}
#     fade_circles = {}
#     frame_idx = 0

#     dummy_image = cv2.resize(black_image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)
#     cv2.imshow("Revealing Image", dummy_image)
#     time.sleep(1)

#     midi, midi_file = prepare_audio_and_midi()
#     pygame.mixer.music.play()
#     start_time = time.time()

#     scale_x = width_first / image.shape[1]
#     scale_y = height_first / image.shape[0]
#     glow_circle = make_glow_circle(GLOW_RADIUS, max_alpha=GLOW_MAX_ALPHA, color=GLOW_COLOR)

#     for msg in midi_file.play():
#         current_time = time.time() - start_time

#         # Fade out glow circles
#         to_remove = []
#         for key in fade_circles:
#             fade_circles[key] += 0.01
#             if fade_circles[key] > FADE_DURATION:
#                 to_remove.append(key)
#         for key in to_remove:
#             fade_circles.pop(key)

#         new_pixels_this_frame = []

#         # Reveal logic
#         if msg.type == 'note_on' and msg.velocity > 0:

#             if msg.velocity <= 20 or random.random() < 0.5:
#                 overlay_active = True
#                 overlay_start_time = time.time()

#             revealed = 0
#             # Count revealed pixels
#             total_pixels = image.shape[0] * image.shape[1]
#             revealed_pixels = np.count_nonzero(np.any(black_image != 0, axis=2))
#             revealed_ratio = revealed_pixels / total_pixels

#             if revealed_ratio <= 0.5 and msg.velocity >= 95 and msg.note < 60:
#                 # Use a set to track unrevealed, non-dark pixels for efficient random selection
#                 if not hasattr(reveal_image_with_music, "unrevealed_pixels"):
#                     reveal_image_with_music.unrevealed_pixels = {
#                     (y, x)
#                     for y in range(image.shape[0])
#                     for x in range(image.shape[1])
#                     if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
#                     }
#                 unrevealed = reveal_image_with_music.unrevealed_pixels
#                 if unrevealed:
#                     choices = list(unrevealed)
#                     np.random.shuffle(choices)
#                     for y, x in choices:
#                         black_image[y, x] = image[y, x]
#                         pulsing_pixels[(y, x)] = 0.0
#                         new_pixels_this_frame.append((y, x))
#                         fade_circles[(y, x)] = 0.0
#                         unrevealed.remove((y, x))
#                         revealed += 1
#                         if revealed >= REVEAL_STEP:
#                             break
#             else:
#                 # High note: reveal from center outward
#                 if msg.note > 72 and ON_HIGH:
#                     center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
#                     unrevealed = [
#                         (y, x)
#                         for y in range(image.shape[0])
#                         for x in range(image.shape[1])
#                         if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
#                     ]
#                     # Sort by distance from center (closest first)
#                     unrevealed.sort(key=lambda p: (p[0] - center_y) ** 2 + (p[1] - center_x) ** 2)
#                     for y, x in unrevealed[:REVEAL_STEP]:
#                         black_image[y, x] = image[y, x]
#                         pulsing_pixels[(y, x)] = 0.0
#                         new_pixels_this_frame.append((y, x))
#                         fade_circles[(y, x)] = 0.0
#                         revealed += 1
#                 # Low note: reveal from edges inward
#                 elif msg.note < 48 and ON_HIGH:
#                     center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
#                     unrevealed = [
#                         (y, x)
#                         for y in range(image.shape[0])
#                         for x in range(image.shape[1])
#                         if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
#                     ]
#                     # Sort by distance from center (farthest first)
#                     unrevealed.sort(key=lambda p: -((p[0] - center_y) ** 2 + (p[1] - center_x) ** 2))
#                     for y, x in unrevealed[:REVEAL_STEP]:
#                         black_image[y, x] = image[y, x]
#                         pulsing_pixels[(y, x)] = 0.0
#                         new_pixels_this_frame.append((y, x))
#                         fade_circles[(y, x)] = 0.0
#                         revealed += 1
#                 # Otherwise: group-based reveal
#                 else:
#                     if group_index < len(pixel_groups):
#                         group = pixel_groups[group_index]
#                         while group_cursor < len(group) and revealed < REVEAL_STEP:
#                             y, x = group[group_cursor]
#                             if not is_dark_pixel(image[y, x]) and np.all(black_image[y, x] == 0):
#                                 black_image[y, x] = image[y, x]
#                                 pulsing_pixels[(y, x)] = 0.0
#                                 new_pixels_this_frame.append((y, x))
#                                 fade_circles[(y, x)] = 0.0
#                                 revealed += 1
#                             group_cursor += 1
#                         if group_cursor >= len(group):
#                             group_index += 1
#                             group_cursor = 0

#         # Update pulsing pixels
#         for (y, x), phase in list(pulsing_pixels.items()):
#             phase += 2 * math.pi * PULSE_SPEED * 0.01
#             pulse = (math.sin(phase) * 0.5 + 0.5) * PULSE_AMPLITUDE + (1 - PULSE_AMPLITUDE)
#             orig_color = image[y, x].astype(np.float32)
#             pulsed_color = np.clip(orig_color * pulse, 0, 255)
#             black_image[y, x] = pulsed_color.astype(np.uint8)
#             pulsing_pixels[(y, x)] = phase

#         # Prepare frame for display and saving
#         big_version = cv2.resize(black_image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)

#         # Overlay glow circles
#         for (y, x), age in fade_circles.items():
#             alpha_factor = max(0, 1 - age / FADE_DURATION)
#             center_x = int(round(x * scale_x + scale_x / 2)) - GLOW_RADIUS
#             center_y = int(round(y * scale_y + scale_y / 2)) - GLOW_RADIUS
#             glow_with_fade = glow_circle.copy()
#             glow_with_fade[..., 3] = (glow_with_fade[..., 3].astype(np.float32) * alpha_factor).astype(np.uint8)
#             overlay_rgba(big_version, glow_with_fade, (center_x, center_y))

#         # Transparent full-image overlay effect
#         if overlay_active:
#             print(1)
#             elapsed = time.time() - overlay_start_time
#             if elapsed <= overlay_duration:
#                 # Blend the full image on top of black_image (gently)
#                 overlay_image = cv2.addWeighted(
#                     big_version, 
#                     1.0, 
#                     cv2.resize(image, (width_first, height_first), interpolation=cv2.INTER_NEAREST), 
#                     overlay_opacity, 
#                     0
#                 )
#                 flash_opacity = 0.15
#                 white_flash = np.full_like(big_version, 255)
#                 overlay_image = cv2.addWeighted(overlay_image, 1.0, white_flash, flash_opacity, 0)
#                 big_version = overlay_image
#             else:
#                 overlay_active = False  # effect ends


#         cv2.imshow("Revealing Image", big_version)
#         frame_path = os.path.join(FRAME_FOLDER, f"frame_{frame_idx:05d}.png")
        
        
#         cv2.imwrite(frame_path, big_version)
#         frame_idx += 1

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# --- MAIN ENTRY POINT ---

if __name__ == "__main__":
    setup_frame_folder()
    reveal_image_with_music()
