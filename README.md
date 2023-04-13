# AI Video Genie
This project showcases the power of LLM's and embedding models in combination with vector databases and some Python magic.

## Getting Started
We use Chroma to store our embedded data in a persistent database. We can add data whenever we want so this is like a
video library you can always rewatch. The unique thing is that you can ask questions about the video and get answers 
based on the video data you have stored in the database. The answers are text based but you'll also get a timestamp
where the answer is located in the video.

Install 
RuntimeError: No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.


Mac `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
`brew install ffmpeg`
