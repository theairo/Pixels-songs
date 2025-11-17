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
IMAGE_PATH = "data/input/image.png"
MIDI_PATH = "data/input/Another Love by Tom Odell.mid"
AUDIO_PATH = "data/input/Another Love by Tom Odell.mp3"
FRAME_FOLDER = "data/output/frames"
OUTPUT_VIDEO = "data/output/output_video.mp4"
START_TIME = 140  # seconds
RANDOM_GROUP_SEED = 21 # 13
FPS = 30
NUM_COLOR_GROUPS = 40 # 14
REVEAL_STEP = 20 # 15
FADE_DURATION = 1.0
GLOW_RADIUS = 10
GLOW_COLOR = (255, 255, 255) # 0 255 200
GLOW_MAX_ALPHA = 0
PULSE_SPEED = 5 # 15 default
PULSE_AMPLITUDE = 0.05 # 0.3 default
ON_HIGH = False
SCALE_IMAGE = 0.47 # 0.2
DISPLAY_SCALE = 0.5

revealed_map = {}

# --- UTILITY FUNCTIONS ---

def is_dark_pixel(pixel, threshold=20):
    luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.5 * pixel[2] # 0.0722
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

def apply_opacity(image, opacity):
        """
        Blend the image with a black background using the given opacity.
        opacity: float in [0, 1], where 1 is fully visible, 0 is fully transparent.
        Returns the blended image.
        """
        black_bg = np.zeros_like(image)
        return cv2.addWeighted(image, opacity, black_bg, 1 - opacity, 0)


def load_and_prepare_images():
    image2 = cv2.imread(IMAGE_PATH)
    height_first, width_first, _ = image2.shape
    image = cv2.resize(image2, (0, 0), fx=SCALE_IMAGE, fy=SCALE_IMAGE, interpolation=cv2.INTER_AREA)
    # Create a grayscale version of the original image as the background
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # Example usage:
    opacity = 0.5  # 10% visible, 90% transparent
    background_image = apply_opacity(background_image, opacity)
    return image, background_image, image2, height_first, width_first

def prepare_audio_and_midi():
    midi = pretty_midi.PrettyMIDI(MIDI_PATH)
    midi_file = mido.MidiFile(MIDI_PATH)
    pygame.mixer.init()
    pygame.mixer.music.load(AUDIO_PATH)
    return midi, midi_file

def setup_frame_folder():
    os.makedirs(FRAME_FOLDER, exist_ok=True)

# --- REVEAL LOGIC ---

def reveal_from_random_cluster(image, background_image, revealed_pixels, pulsing_pixels, fade_circles):
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
        if (y, x) not in revealed_map and not is_dark_pixel(image[y, x])
    ]

    np.random.shuffle(unrevealed)
    unrevealed.sort(key=lambda p: ((p[0]-center_y)**2 + (p[1]-center_x)**2) * (1 + 0.2 * np.random.rand()))

    return reveal_pixels(unrevealed[:REVEAL_STEP*2], image, background_image, pulsing_pixels, fade_circles)

def reveal_from_random_global_cache(image, background_image, REVEAL_STEP_MULTIPLIED, pulsing_pixels, fade_circles):
    if not hasattr(reveal_from_random_global_cache, "unrevealed_pixels"):
        reveal_from_random_global_cache.unrevealed_pixels = {
            (y, x)
            for y in range(image.shape[0])
            for x in range(image.shape[1])
            if (y, x) not in revealed_map and not is_dark_pixel(image[y, x])
        }

    unrevealed = reveal_from_random_global_cache.unrevealed_pixels
    if not unrevealed:
        return []

    choices = list(unrevealed)
    np.random.shuffle(choices)

    new_pixels = []
    revealed = 0
    for y, x in choices:
        if (y, x) not in revealed_map:
            background_image[y, x] = image[y, x]
            revealed_map[(y, x)] = True
            pulsing_pixels[(y, x)] = 0.0
            fade_circles[(y, x)] = (0.0, GLOW_COLOR)
            new_pixels.append((y, x))
            unrevealed.remove((y, x))
            revealed += 1
            if revealed >= REVEAL_STEP_MULTIPLIED:
                break

    return new_pixels

def reveal_pixels(pixels, image, background_image, pulsing_pixels, fade_circles):
    revealed = []
    for y, x in pixels:
        if (y, x) not in revealed_map:  # not already revealed
            background_image[y, x] = image[y, x]
            revealed_map[(y, x)] = True
            pulsing_pixels[(y, x)] = 0.0
            fade_circles[(y, x)] = (0.0, GLOW_COLOR)
            revealed.append((y, x))
    return revealed

def reveal_from_center_out(image, background_image, REVEAL_STEP, pulsing_pixels, fade_circles):
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2

    unrevealed = [
        (y, x)
        for y in range(h)
        for x in range(w)
        if (y, x) not in revealed_map and not is_dark_pixel(image[y, x])
    ]
    unrevealed.sort(key=lambda p: (p[0]-center_y)**2 + (p[1]-center_x)**2)

    return reveal_pixels(unrevealed[:REVEAL_STEP], image, background_image, pulsing_pixels, fade_circles)

def reveal_from_edges_in(image, background_image, REVEAL_STEP, pulsing_pixels, fade_circles):
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2

    unrevealed = [
        (y, x)
        for y in range(h)
        for x in range(w)
        if (y, x) not in revealed_map and not is_dark_pixel(image[y, x])
    ]
    unrevealed.sort(key=lambda p: -((p[0]-center_y)**2 + (p[1]-center_x)**2))

    return reveal_pixels(unrevealed[:REVEAL_STEP], image, background_image, pulsing_pixels, fade_circles)

def reveal_grouped(group, group_cursor, REVEAL_STEP_MULTIPLIED, image, background_image, pulsing_pixels, fade_circles):
    revealed = 0
    new_pixels = []
    while group_cursor < len(group) and revealed < REVEAL_STEP_MULTIPLIED:
        y, x = group[group_cursor]
        if not is_dark_pixel(image[y, x]) and (y, x) not in revealed_map:
            background_image[y, x] = image[y, x]
            revealed_map[(y, x)] = True
            pulsing_pixels[(y, x)] = 0.0
            fade_circles[(y, x)] = (0.0, GLOW_COLOR)
            new_pixels.append((y, x))
            revealed += 1
        group_cursor += 1
    return new_pixels, group_cursor

def reveal_grouped_reverse(group, group_cursor, image, background_image, pulsing_pixels, fade_circles):
    revealed = 0
    new_pixels = []
    while group_cursor >= 0 and revealed < REVEAL_STEP:
        y, x = group[group_cursor]
        if not is_dark_pixel(image[y, x]) and (y, x) not in revealed_map:
            background_image[y, x] = image[y, x]
            revealed_map[(y, x)] = True
            pulsing_pixels[(y, x)] = 0.0
            fade_circles[(y, x)] = (0.0, GLOW_COLOR)
            new_pixels.append((y, x))
            revealed += 1
        group_cursor -= 1
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



def update_bee(image, background_image, pulsing_pixels, fade_circles, video_time, bee):
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
                if (ny, nx) not in revealed_map and not is_dark_pixel(image[ny, nx]):
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
        if (cy, cx) not in revealed_map and not is_dark_pixel(image[cy, cx]):
            background_image[cy, cx] = image[cy, cx]
            revealed_map[(y, x)] = True
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


import subprocess

TRIMMED_AUDIO = "trimmed_audio.mp3"


subprocess.run([
    "ffmpeg", "-y",
    "-ss", str(START_TIME),
    "-i", AUDIO_PATH,
    "-acodec", "copy",
    TRIMMED_AUDIO
])

def calculate_scale_from_reveal_step(image2, midi_file, seconds):
    """
    Calculates SCALE_IMAGE so that the image size matches the desired REVEAL_STEP per note
    for notes played in the first `seconds` seconds of the MIDI file.
    """
    # Count note_on events with velocity > 0 within the time window
    current_time_sec = 0
    note_on_count = 0
    for msg in midi_file:
        current_time_sec += msg.time
        if current_time_sec > seconds:
            break
        if msg.type == 'note_on' and msg.velocity > 0:
            note_on_count += 1
    if note_on_count == 0:
        note_on_count = 1
    desired_total_pixels = REVEAL_STEP * note_on_count
    h, w, _ = image2.shape
    orig_total_pixels = h * w
    scale = np.sqrt(desired_total_pixels / orig_total_pixels)
    print(f"SCALE_IMAGE for first {seconds}s: {scale}")
    return scale

def build_note_hand_lookup(midi_path):
    """
    Returns a dict mapping (start_time, pitch) to instrument index and name.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    note_hand = {}
    for i, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            # Use rounded start time for easier matching
            key = (round(note.start, 3), note.pitch)
            note_hand[key] = i # 0 - right, 1 - left
    return note_hand

def find_hand_info(note_hand_lookup, video_time, pitch, tolerance=0.5):
    for (start_time, note_pitch), info in note_hand_lookup.items():
        if note_pitch == pitch and abs(start_time - video_time) < tolerance:
            return info
    return None

def get_reveal_count_from_velocity(velocity, min_velocity=20, max_velocity=127, multiplier_range=(0.5, 2.0)):

    # Clamp velocity within the expected range
    velocity = max(min_velocity, min(max_velocity, velocity))

    # Normalize velocity between 0 and 1
    norm_vel = (velocity - min_velocity) / (max_velocity - min_velocity)

    # Map to multiplier range
    min_mult, max_mult = multiplier_range
    multiplier = min_mult + norm_vel * (max_mult - min_mult)

    # Compute and return final pixel count
    return int(REVEAL_STEP * multiplier)

# --- MAIN REVEAL LOOP ---
def reveal_image_with_music(mode="render", full_song=False):
    """
    mode: "render" for video+audio output, "view" for slow preview (no video file written)
    """
    
    #score.parts[0].measures(13, 14).show('text')

    note_hand_lookup = build_note_hand_lookup(MIDI_PATH)

    overlays = []
    glow_active = False
    glow_start_time = 0

    # Initialize prev_velocity attribute for accented note detection
    reveal_image_with_music.prev_velocity = None

    image, background_image, image2, height_first, width_first = load_and_prepare_images()
    pixel_groups = group_pixels_by_color(image, num_groups=NUM_COLOR_GROUPS)

    # --- Randomize group sequence with a seed ---
    
    random.seed(RANDOM_GROUP_SEED)
    random.shuffle(pixel_groups)

    group_index = 0
    group_index_left = len(pixel_groups) - 1
    group_cursor = 0
    group_cursor_left = len(pixel_groups[group_index_left])-1
    pulsing_pixels = {}
    fade_circles = {}
    frame_idx = 0

    midi, midi_file = prepare_audio_and_midi()

    calculate_scale_from_reveal_step(image2, midi_file, 58)
    # Calculate REVEAL_STEP for full song mode
    if full_song:
        total_pixels = image.shape[0] * image.shape[1]
        # Count all note_on events with velocity > 0
        note_on_count = sum(1 for msg in midi_file if msg.type == 'note_on' and msg.velocity > 0)
        global REVEAL_STEP
        REVEAL_STEP = max(1, int(np.floor(total_pixels / note_on_count)))
        print(f"[Full Song Mode] Calculated REVEAL_STEP: {REVEAL_STEP} (pixels per note)")




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

    
    # Skip all events before START_TIME
    event_idx = 0

    total_events = len(midi_events)
    while event_idx < total_events and midi_events[event_idx][0] < START_TIME:
        event_idx += 1

    print(image.shape[:2])


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
        preview_start_time = time.time() - START_TIME
        
    cnt = 0
    # Main loop: generate frames based on MIDI timing, not wall clock
    video_time = START_TIME
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

                
                hand = find_hand_info(note_hand_lookup, video_time, msg.note)


                if not hand:
                    print(f"Note {msg.note} hand unknown at {video_time:.2f}s")

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
                total_pixels = len([
                    (y, x)
                    for y in range(image.shape[0])
                    for x in range(image.shape[1])
                    if not is_dark_pixel(image[y, x])
                ])
                revealed_pixels = len(revealed_map)
                revealed_ratio = revealed_pixels / total_pixels if total_pixels > 0 else 0

                REVEAL_STEP_MULTIPLIED = get_reveal_count_from_velocity(msg.velocity)



                if False:
                    for bee in BUMBLEBEES:
                        # Append current note timestamp to this bee's note_times
                        bee["note_times"].append(video_time)

                        # Remove timestamps older than 0.5 seconds
                        bee["note_times"] = [t for t in bee["note_times"] if video_time - t <= 0.5]

                        # Update this bee (reveal cluster etc)
                        new_pixels = update_bee(image, background_image, pulsing_pixels, fade_circles, video_time, bee)

                elif msg.note > 120:
                    new_pixels = reveal_from_random_cluster(image, background_image, revealed_pixels, pulsing_pixels, fade_circles)

                elif hand==1 or msg.note<60:
                    new_pixels = reveal_from_random_global_cache(image, background_image, REVEAL_STEP_MULTIPLIED, pulsing_pixels, fade_circles)

                elif msg.note > 72 and ON_HIGH:
                    new_pixels = reveal_from_center_out(image, background_image, pulsing_pixels, fade_circles)

                elif msg.note < 48 and ON_HIGH:
                    new_pixels = reveal_from_edges_in(image, background_image, pulsing_pixels, fade_circles)

                elif hand == 1:
                    if group_index_left >= 0:
                        group = pixel_groups[group_index_left]
                        new_pixels, group_cursor_left = reveal_grouped_reverse(group, group_cursor_left, image, background_image, pulsing_pixels, fade_circles)
                        if group_cursor_left < 0:
                            group_index_left -= 1
                            if group_index_left >= 0:
                                group_cursor_left = len(pixel_groups[group_index_left]) - 1
                else:
                    if group_index < len(pixel_groups):
                        group = pixel_groups[group_index]
                        new_pixels, group_cursor = reveal_grouped(group, group_cursor, REVEAL_STEP_MULTIPLIED, image, background_image, pulsing_pixels, fade_circles)
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
            background_image[y, x] = pulsed_color.astype(np.uint8)
            pulsing_pixels[(y, x)] = phase

        # Prepare frame for display and saving
        big_version = cv2.resize(background_image, (width_first, height_first), interpolation=cv2.INTER_NEAREST)
        # Window size
        display_frame = cv2.resize(big_version, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
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

            # Overlay glow on any background (not just black)
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



        
        # Fade out opacity for background image in first 3 seconds

        BG_FADE_DURATION = 3.0
        start_opacity = 0.5
        end_opacity = 0.1

        if video_time < START_TIME + BG_FADE_DURATION:
            fade_progress = (video_time - START_TIME) / BG_FADE_DURATION
            fade_progress = max(0.0, min(1.0, fade_progress))
            fade_opacity = start_opacity - fade_progress * (start_opacity - end_opacity)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            background_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            
            background_image = apply_opacity(background_image, fade_opacity)
        else:
            fade_opacity = end_opacity
            
        # Show current time in view mode
        print(f"Current time: {video_time:.2f} seconds")

        cv2.imshow("Revealing Image", display_frame)
        if mode == "render":
            cv2.imwrite(f"{FRAME_FOLDER}/frame_{frame_idx:05d}.png", big_version)
            video_writer.write(big_version)

        frame_idx += 1

        if (video_time - START_TIME) >= 60:
             quit_flag = True
             break

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
            "-i", TRIMMED_AUDIO,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            OUTPUT_VIDEO
        ]
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
