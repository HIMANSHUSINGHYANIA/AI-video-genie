import random
from typing import List, Dict, Any

from pydantic.typing import StrPath
from pytube import YouTube
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import openai


class Video:
    """
    Wrapper of the pytube library for the specific use case of the project
    """
    def __init__(self, url: str):
        self.url = url
        self.yt = YouTube(url)
        self.stream = self.yt.streams.filter(file_extension='mp4')
        self.output_dir = "files"
        self.chunks_dir = "chunks"

    def download(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.video = self.stream.first().download(output_path=self.output_dir)
        return self

    def chunks(self, chunk_duration_in_sec: int = 30):
        chunks = []
        if not os.path.exists(self.chunks_dir):
            os.makedirs(self.chunks_dir)
        video_clip = VideoFileClip(self.video)

        total_duration = video_clip.duration

        num_chunks = int(total_duration // chunk_duration_in_sec) + 1

        for i in range(num_chunks):
            # calculate the start and end times for the chunk
            start_time = i * chunk_duration_in_sec
            end_time = min((i + 1) * chunk_duration_in_sec, total_duration)
            # extract the chunk from the video clip
            chunk = video_clip.subclip(start_time, end_time)

            # Change path to chunks
            filename = f"{self.video}-{i}.mp4".replace('/files/', '/chunks/')
            chunk.write_videofile(filename, threads=4, codec="libx264", logger=None)
            chunks.append(filename)

        video_clip.close()
        return chunks

    @staticmethod
    def transcribe(files: List[StrPath]) -> List[Dict[str, Any]]: # | StrPath):
        transcripts = []
        if not isinstance(files, list):
            files = [files]
        for sec, path in enumerate(files):
            f = open(path, "rb")
            transcript = openai.Audio.transcribe("whisper-1", file=f)
            # replace chunks with transcript
            with open(f"{path}.txt".replace('chunks', 'transcript'), "w") as f:
                f.write(transcript["text"])
                f.close()

                # we split to 30 seconds batches
                transcripts.append({"path": f"{path}.txt".replace('chunks', 'transcript'),
                                    "min": {"sec": sec * 30},
                                    "transcript": transcript["text"], "id": str(random.randint(0, 9999999999))})
        return transcripts

    @staticmethod
    def merge_transcriptions(transcriptions: List[StrPath]) -> StrPath:
        full_doc = ""
        name = transcriptions[0]["path"].split(".")[0].split("/")[-1]
        for t in transcriptions:
            doc = open(t["path"], "r")
            full_doc += doc.read()
            doc.close()
        with open(f"transcript/{name}-full.txt", "w") as f:
            f.write(full_doc)
            f.close()
        return f"transcript/{name}-full.txt"

if __name__ == "__main__":
    video = Video("https://www.youtube.com/watch?v=nQ2A30cD3Q8&t=15s&ab_channel=Fireship")
    video.download()
    chunks = video.chunks()
    video.transcribe(chunks)
