# AI Video Genie
This project showcases the power of LLM's and embedding models in combination with vector databases and some Python magic.
The goal is to create a video library that can be queried with questions and answers are returned based on the video data.

## Getting Started
We use Chroma to store our embedded data in a persistent database. We can add data whenever we want so this is like a
video library you can always rewatch. The unique thing is that you can ask questions about the video and get answers 
based on the video data you have stored in the database. The answers are text based but you'll also get a timestamp
where the answer is located in the video.

## Installation

`pip install -r requirements.txt`

You also need to install ffmpeg. 
On Windows you can download it from [here](https://ffmpeg.org/download.html). On Mac you can use Homebrew:

Install homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

Install ffmpeg: `brew install ffmpeg`

## Usage

In the `main.py` file you specify a youtube-video in the `video_url` variable. 
The `n_results` parameter specifies how many starting points on the video you want to get returned.
This is based on the closest neighbors in the vector database. 
So the less results you specify, the more accurate the results will be.


## Video

[![Screenshot](https://fastupload.io/secure/file/v1RgzRv99zbpB)](https://youtu.be/mhdOTLp-IjQ)