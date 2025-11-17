# Pixels-songs

## Project Description

This project explores algorithmic composition by generating musical sequences directly from image pixel data. It reads the pixels of an image and translates their color or brightness values into musical notes, creating a unique "song" for any given image.

This was built as a personal project to learn more about creative AI applications and generative art.

Demo Video

Click the thumbnail below to see (and hear) an example of the output:

## Technologies Used

- Python

- OpenCV (cv2): For reading and processing image pixel data.

- NumPy: For numerical operations and array manipulation.

- Scikit-learn (sklearn): Used for K-Means clustering to analyze image color palettes.

- Mido & Pretty MIDI: For creating, manipulating, and saving MIDI music files.

- Pygame: For audio playback.

## How to Run

Clone the repository:

git clone [https://github.com/theairo/Pixels-songs.git](https://github.com/theairo/Pixels-songs.git)
cd Pixels-songs


Install the necessary libraries:

pip install -r requirements.txt


Run the main script from the root folder:

python src/main.py


(Note: The input and output files are set inside src/main.py)

## Project Status

Finished project.
