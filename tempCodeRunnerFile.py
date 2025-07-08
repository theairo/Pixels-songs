# Prepare frame for display and saving
        # if overlay_active:
        #     elapsed = time.time() - overlay_start_time
        #     if elapsed <= overlay_duration:
        #         # Blend the full image on top of black_image (gently)
        #         overlay_image = cv2.addWeighted(big_version, 1.0, cv2.resize(image2, (width_first, height_first)), overlay_opacity, 0)
        #         big_version = overlay_image
        #     else:
        #         overlay_active = False  # effect ends