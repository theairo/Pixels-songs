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
MIDI_PATH = "Vivaldi_-_Violin_Concerto_in_F_minor_Op._8_No._4_RV._297_Winter_for_Solo_Piano.mid"
AUDIO_PATH = "Vivaldi_-_Violin_Concerto_in_F_minor_Op._8_No._4_RV._297_Winter_for_Solo_Piano.mp3"
FRAME_FOLDER = "frames"
OUTPUT_VIDEO = "output_video.mp4"
FPS = 60
NUM_COLOR_GROUPS = 16
REVEAL_STEP = 8 # 8 for 
FADE_DURATION = 1.0
GLOW_RADIUS = 10
GLOW_COLOR = (0, 255, 200)
GLOW_MAX_ALPHA = 0.8
PULSE_SPEED = 0.5 # 15 default
PULSE_AMPLITUDE = 0.05 # 0.3 default
ON_HIGH = False
SCALE_IMAGE = 0.18 # 0.12

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

# --- REVEAL LOGIC ---

def reveal_from_random_cluster(image, black_image, revealed_pixels, REVEAL_STEP, pulsing_pixels, fade_circles):
    # Pick a new center for every note
    h, w = image.shape[:2]
    spread = 0.3 + 0.4 * abs(revealed_pixels / (h * w) - 0.5)
    jitter_x = random.uniform(-0.15, 0.15)
    jitter_y = random.uniform(-0.15, 0.15)

    center_x = random.randint(int(w * max(0.05, 0.5 - spread/2 + jitter_x)),
                              int(w * min(0.95, 0.5 + spread/2 + jitter_x)))
    center_y = random.randint(int(h * max(0.05, 0.5 - spread/2 + jitter_y)),
                              int(h * min(0.95, 0.5 + spread/2 + jitter_y)))

    unrevealed = [
        (y, x)
        for y in range(h)
        for x in range(w)
        if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
    ]

    np.random.shuffle(unrevealed)
    unrevealed.sort(key=lambda p: ((p[0]-center_y)**2 + (p[1]-center_x)**2) * (1 + 0.2 * np.random.rand()))

    return reveal_pixels(unrevealed[:REVEAL_STEP*2], image, black_image, pulsing_pixels, fade_circles)

def reveal_from_random_global_cache(image, black_image, REVEAL_STEP, pulsing_pixels, fade_circles):
    if not hasattr(reveal_from_random_global_cache, "unrevealed_pixels"):
        reveal_from_random_global_cache.unrevealed_pixels = {
            (y, x)
            for y in range(image.shape[0])
            for x in range(image.shape[1])
            if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
        }

    unrevealed = reveal_from_random_global_cache.unrevealed_pixels
    if not unrevealed:
        return []

    choices = list(unrevealed)
    np.random.shuffle(choices)

    new_pixels = []
    revealed = 0
    for y, x in choices:
        if np.all(black_image[y, x] == 0):
            black_image[y, x] = image[y, x]
            pulsing_pixels[(y, x)] = 0.0
            fade_circles[(y, x)] = (0.0, GLOW_COLOR)
            new_pixels.append((y, x))
            unrevealed.remove((y, x))
            revealed += 1
            if revealed >= REVEAL_STEP:
                break

    return new_pixels

def reveal_pixels(pixels, image, black_image, pulsing_pixels, fade_circles):
    revealed = []
    for y, x in pixels:
        if np.all(black_image[y, x] == 0):  # not already revealed
            black_image[y, x] = image[y, x]
            pulsing_pixels[(y, x)] = 0.0
            fade_circles[(y, x)] = (0.0, GLOW_COLOR)
            revealed.append((y, x))
    return revealed

def reveal_from_center_out(image, black_image, REVEAL_STEP, pulsing_pixels, fade_circles):
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2

    unrevealed = [
        (y, x)
        for y in range(h)
        for x in range(w)
        if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
    ]
    unrevealed.sort(key=lambda p: (p[0]-center_y)**2 + (p[1]-center_x)**2)

    return reveal_pixels(unrevealed[:REVEAL_STEP], image, black_image, pulsing_pixels, fade_circles)

def reveal_from_edges_in(image, black_image, REVEAL_STEP, pulsing_pixels, fade_circles):
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2

    unrevealed = [
        (y, x)
        for y in range(h)
        for x in range(w)
        if np.all(black_image[y, x] == 0) and not is_dark_pixel(image[y, x])
    ]
    unrevealed.sort(key=lambda p: -((p[0]-center_y)**2 + (p[1]-center_x)**2))

    return reveal_pixels(unrevealed[:REVEAL_STEP], image, black_image, pulsing_pixels, fade_circles)

def reveal_grouped(group, group_cursor, REVEAL_STEP, image, black_image, pulsing_pixels, fade_circles):
    revealed = 0
    new_pixels = []
    while group_cursor < len(group) and revealed < REVEAL_STEP:
        y, x = group[group_cursor]
        if not is_dark_pixel(image[y, x]) and np.all(black_image[y, x] == 0):
            black_image[y, x] = image[y, x]
            pulsing_pixels[(y, x)] = 0.0
            fade_circles[(y, x)] = (0.0, GLOW_COLOR)
            new_pixels.append((y, x))
            revealed += 1
        group_cursor += 1
    return new_pixels, group_cursor

NUM_BEES = 0  # or however many bees you want initially

BUMBLEBEES = [
    {
        "pos": (150, 150),
        "velocity": (0.0, 0.0),
        "note_times": [],
        "glow_color": (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
    }
    for _ in range(NUM_BEES)
]



def update_bee(image, black_image, pulsing_pixels, fade_circles, video_time, bee):
    h, w = image.shape[:2]
    pos_y, pos_x = bee["pos"]
    vel_y, vel_x = bee["velocity"]
    note_times = bee["note_times"]
    color = bee["glow_color"]

    note_density = len(note_times) / 0.5
    speed = min(7.0, 0.5 + note_density * 0.3)

    search_radius = 20
    candidates = []
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            ny = pos_y + dy
            nx = pos_x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if np.all(black_image[ny, nx] == 0) and not is_dark_pixel(image[ny, nx]):
                    candidates.append((ny, nx))

    if candidates:
        target_y, target_x = random.choice(candidates)
        dir_y = target_y - pos_y
        dir_x = target_x - pos_x
        norm = math.sqrt(dir_y**2 + dir_x**2) + 1e-5
        dir_y /= norm
        dir_x /= norm

        jitter_angle = random.uniform(-math.pi / 6, math.pi / 6)
        angle = math.atan2(dir_y, dir_x) + jitter_angle
        vel_x = speed * math.cos(angle)
        vel_y = speed * math.sin(angle)
    else:
        vel_y = random.uniform(-1, 1) * speed
        vel_x = random.uniform(-1, 1) * speed

    new_y = int(pos_y + vel_y)
    new_x = int(pos_x + vel_x)
    new_y = max(0, min(h - 1, new_y))
    new_x = max(0, min(w - 1, new_x))

    fade_circles[(new_y, new_x)] = (0.0, color)

    bee["pos"] = (new_y, new_x)
    bee["velocity"] = (vel_y, vel_x)

    cluster = []
    for _ in range(4):
        dy = random.randint(-1, 1)
        dx = random.randint(-1, 1)
        cy = min(max(0, new_y + dy), h - 1)
        cx = min(max(0, new_x + dx), w - 1)
        if np.all(black_image[cy, cx] == 0) and not is_dark_pixel(image[cy, cx]):
            black_image[cy, cx] = image[cy, cx]
            pulsing_pixels[(cy, cx)] = 0.0
            fade_circles[(cy, cx)] = (0.0, color)
            cluster.append((cy, cx))

    return cluster


def make_white_glow_mask(radius, max_alpha):
    size = radius * 2 + 1
    glow = np.zeros((size, size, 4), dtype=np.uint8)
    
    center = radius
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
            fade = max(0, 1 - dist / radius)
            alpha = int(max_alpha * 255 * fade)
            glow[y, x] = [255, 255, 255, alpha]  # White RGB, fading alpha
    return glow

BEES = []

LAST_BEE_ADD_TIME = -float('inf')
BEE_ADD_INTERVAL = 5.0  # seconds

def try_add_bee(video_time, height, width):
    global LAST_BEE_ADD_TIME
    if video_time - LAST_BEE_ADD_TIME >= BEE_ADD_INTERVAL:
        LAST_BEE_ADD_TIME = video_time
        new_bee = {
            "pos": (random.randint(0, height - 1), random.randint(0, width - 1)),
            "velocity": (0.0, 0.0),
            "note_times": [],
            "glow_color": (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)),
        }
        BUMBLEBEES.append(new_bee)

# --- MAIN REVEAL LOOP ---
def reveal_image_with_music(mode="render", full_song=False):
    """
    mode: "render" for video+audio output, "view" for slow preview (no video file written)
    """
    
    #score.parts[0].measures(13, 14).show('text')

    overlays = []
    glow_active = False
    glow_start_time = 0

    # Initialize prev_velocity attribute for accented note detection
    reveal_image_with_music.prev_velocity = None

    image, black_image, image2, height_first, width_first = load_and_prepare_images()
    pixel_groups = group_pixels_by_color(image, num_groups=NUM_COLOR_GROUPS)

    # --- Randomize group sequence with a seed ---
    RANDOM_GROUP_SEED = 36  # Change this for different random orders
    random.seed(RANDOM_GROUP_SEED)
    random.shuffle(pixel_groups)

    group_index = 0
    group_cursor = 0
    pulsing_pixels = {}
    fade_circles = {}
    frame_idx = 0

    # Calculate REVEAL_STEP for full song mode
    if full_song:
        total_pixels = image.shape[0] * image.shape[1]
        # Count all note_on events with velocity > 0
        midi, midi_file = prepare_audio_and_midi()
        note_on_count = sum(1 for msg in midi_file if msg.type == 'note_on' and msg.velocity > 0)
        global REVEAL_STEP
        REVEAL_STEP = max(1, int(np.floor(total_pixels / note_on_count)))
        print(f"[Full Song Mode] Calculated REVEAL_STEP: {REVEAL_STEP} (pixels per note)")
    else:
        midi, midi_file = prepare_audio_and_midi()

    dummy_image = cv2.resize(black_image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Revealing Image", dummy_image)
    time.sleep(1)

    # Do NOT start pygame audio playback here; let ffmpeg handle audio sync in render mode
    scale_x = width_first / image.shape[1]
    scale_y = height_first / image.shape[0]
    glow_mask = make_white_glow_mask(GLOW_RADIUS, GLOW_MAX_ALPHA)

    # Set up video writer (only in render mode)
    if mode == "render":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = "temp_video_no_audio.mp4"
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, FPS, (width_first, height_first))
    else:
        video_writer = None

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

    note_history = []

    # Use MIDI duration for video length
    if midi_events:
        midi_total_time = midi_events[-1][0]
    else:
        midi_total_time = 0

    # In view mode, play audio for preview
    if mode == "view":
        pygame.mixer.music.play()
        preview_start_time = time.time()

    # Main loop: generate frames based on MIDI timing, not wall clock
    video_time = 0.0
    while video_time < midi_total_time and not quit_flag:
        # In view mode, sync to real time (slow preview)
        if mode == "view":
            elapsed = time.time() - preview_start_time
            if elapsed < video_time:
                time.sleep(video_time - elapsed)

        h, w = image.shape[:2]
        try_add_bee(video_time, w, h)

        # Process all MIDI messages up to current video_time
        while event_idx < total_events and midi_events[event_idx][0] <= video_time:
            msg_time, msg = midi_events[event_idx]
            event_idx += 1

            new_pixels_this_frame = []

            if msg.type == 'note_on' and msg.velocity > 0:
                # Track note history (for trend detection)
                note_history.append(msg.note)
                if len(note_history) > 8:  # window size, adjust as needed
                    note_history.pop(0)

                # Only add overlay for ~40% of notes, and only if less than 17 seconds into the song
                if random.random() < 0 and video_time < 17:
                    overlays.append({
                        'start_time': video_time,
                        'duration': 0.5,  # seconds
                        'opacity': 0.01   # 1% transparency
                    })

                revealed = 0
                total_pixels = image.shape[0] * image.shape[1]
                revealed_pixels = np.count_nonzero(np.any(black_image != 0, axis=2))

                revealed_ratio = revealed_pixels / total_pixels


                if False:
                    for bee in BUMBLEBEES:
                        # Append current note timestamp to this bee's note_times
                        bee["note_times"].append(video_time)

                        # Remove timestamps older than 0.5 seconds
                        bee["note_times"] = [t for t in bee["note_times"] if video_time - t <= 0.5]

                        # Update this bee (reveal cluster etc)
                        new_pixels = update_bee(image, black_image, pulsing_pixels, fade_circles, video_time, bee)

                elif msg.note > 72:
                    new_pixels = reveal_from_random_cluster(image, black_image, revealed_pixels, REVEAL_STEP, pulsing_pixels, fade_circles)

                elif revealed_ratio <= 0.99 and msg.velocity >= 60 and msg.note < 0:
                    new_pixels = reveal_from_random_global_cache(image, black_image, REVEAL_STEP, pulsing_pixels, fade_circles)

                elif msg.note > 72 and ON_HIGH:
                    new_pixels = reveal_from_center_out(image, black_image, REVEAL_STEP, pulsing_pixels, fade_circles)

                elif msg.note < 48 and ON_HIGH:
                    new_pixels = reveal_from_edges_in(image, black_image, REVEAL_STEP, pulsing_pixels, fade_circles)

                else:
                    if group_index < len(pixel_groups):
                        group = pixel_groups[group_index]
                        new_pixels, group_cursor = reveal_grouped(group, group_cursor, REVEAL_STEP, image, black_image, pulsing_pixels, fade_circles)
                        if group_cursor >= len(group):
                            group_index += 1
                            group_cursor = 0

                new_pixels_this_frame.extend(new_pixels)

        # Update pulsing pixels
        for (y, x), phase in list(pulsing_pixels.items()):
            phase += 2 * math.pi * PULSE_SPEED * frame_duration
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
            age, color = fade_circles[key]
            age += frame_duration
            if age > FADE_DURATION:
                to_remove.append(key)
            else:
                fade_circles[key] = (age, color)  # Update with new age

        for key in to_remove:
            fade_circles.pop(key)

        for (y, x), (age, color) in fade_circles.items():
            alpha_factor = max(0, 1 - age / FADE_DURATION)
            center_x = int(round(x * scale_x + scale_x / 2)) - GLOW_RADIUS
            center_y = int(round(y * scale_y + scale_y / 2)) - GLOW_RADIUS

            glow_faded = np.zeros_like(glow_mask)

            # Recolor RGB channels
            for c in range(3):
                glow_faded[..., c] = (glow_mask[..., 3].astype(np.float32) * color[c] / 255).astype(np.uint8)

            # Fade alpha channel
            glow_faded[..., 3] = (glow_mask[..., 3].astype(np.float32) * alpha_factor).astype(np.uint8)

            overlay_rgba(big_version, glow_faded, (center_x, center_y))

        # Transparent full-image overlay effects (multiple, async)
        overlay_img = cv2.resize(image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)
        overlays_to_remove = []
        for overlay in overlays:
            elapsed_overlay = video_time - overlay['start_time']
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

        # Show current time in view mode
        print(f"Current time: {video_time:.2f} seconds")

        cv2.imshow("Revealing Image", big_version)
        if mode == "render":
            cv2.imwrite(f"{FRAME_FOLDER}/frame_{frame_idx:05d}.png", big_version)
            video_writer.write(big_version)

        frame_idx += 1

        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            quit_flag = True
            break

        video_time += frame_duration

    cv2.destroyAllWindows()
    if mode == "render" and video_writer is not None:
        video_writer.release()
        # --- Combine video and audio using ffmpeg ---
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
        os.remove(temp_video_path)
        # --- Delete all frames after making the video ---
        time.sleep(0.5)
        for fname in os.listdir(FRAME_FOLDER):
            fpath = os.path.join(FRAME_FOLDER, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
        
        print(f"All frames in '{FRAME_FOLDER}' have been deleted.")
    elif mode == "view":
        pygame.mixer.music.stop()

if __name__ == "__main__":
    setup_frame_folder()
    reveal_image_with_music()
