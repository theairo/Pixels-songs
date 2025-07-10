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