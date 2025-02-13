from fer import FER
import cv2
from fer.classes import Video
from matplotlib import pyplot as plt


# Resources to learn how to write this code came from FER documentation and examples on GitHub

def analyzeVideo(video: Video, detector: FER):
    emotions = video.analyze(detector, display=False)

    df = video.to_pandas(emotions)
    df = video.get_first_face(df)
    df = video.get_emotions(df)

    df.plot()
    plt.show()



detector = FER()
video1 = Video("assets/Video_One.mp4")
video2 = Video("assets/Video_Two.mp4")

analyzeVideo(video1, detector)
analyzeVideo(video2, detector)
