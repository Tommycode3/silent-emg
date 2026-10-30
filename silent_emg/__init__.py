"""Silent-speech decoding from facial and neck surface EMG."""

__version__ = "0.1.0"

N_CHANNELS = 8
FS = 1000  # Hz, sampling rate of the Gaddy & Klein (2020) recordings

# Electrode sites from Gaddy & Klein (2020), Table 3 (0-based index here).
CHANNEL_SITES = [
    "left cheek above mouth",
    "left corner of chin",
    "below chin, back 3 cm",
    "throat, 3 cm left of Adam's apple",
    "mid-jaw right",
    "right cheek below mouth",
    "right cheek, 2 cm from nose",
    "back of right cheek, 4 cm before ear",
]
CHANNEL_SHORT = ["L cheek", "L chin", "under chin", "throat", "R jaw", "R cheek low", "R cheek nose", "R cheek back"]
NECK_CHANNELS = [2, 3]  # submental + throat
FACE_CHANNELS = [0, 1, 4, 5, 6, 7]
