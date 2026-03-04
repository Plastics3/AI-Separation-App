AI Project
===========

This repository an AI separation project with first piano separation!

it contain:
1. App with ai music separation
2. instructions and tools for anybody that wants to separate other kind of music

*if you have a laptop look at the bottom

Project Structure (important)

Make sure your project looks like this after cloning:

project-root/
│
├─ app/
│  ├─ app.py
│  └─ PianoSeparationModelWeights/
│
├─ requirements.txt
└─ README.md

* The folder PianoSeparationModelWeights must be inside the app folder.
* The ANN folder is unrelated however is the model I built for other things.

Requirements:

Git

FFmpeg (required for audio loading)

Install FFmpeg

Windows:
Download from https://ffmpeg.org
 and add it to your PATH

macOS:

brew install ffmpeg

Linux:

sudo apt install ffmpeg
Setup Instructions
1️. Clone the repository
command: git clone https://github.com/Plastics3/AI-Separation-App
cd your-repo-name

2. Create a virtual environment (recommended)
python -m venv .venv

Activate it:

Windows

.venv\Scripts\activate

macOS / Linux

source .venv/bin/activate
3️. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

   
4. Run the App

From the project root directory, run:

python app/app.py

The GUI should open.
You can now:

Load an audio file:
Any song/audio that is downloaded

Enable/disable stems:
You can activate/deactivate the stems by clicking the buttons.
You can see what stems are active in the top left.
*it takes time for the effects to take place

Adjust volume:
Bottom right is the app volume button.
each stem has a volume button however it takes time for it to take effect.


* If you have a laptop you may have to download a few files and place them in: 
C:\Users\yourUserName\.cache\torch\hub\checkpoints

Notes:

credit to maestro piano dataset:
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
  Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling
  Factorized Piano Music Modeling and Generation with the MAESTRO Dataset."
  In International Conference on Learning Representations, 2019.

credit to openunmix:
@article{stoter19,  
  author={F.-R. St\\"oter and S. Uhlich and A. Liutkus and Y. Mitsufuji},  
  title={Open-Unmix - A Reference Implementation for Music Source Separation},  
  journal={Journal of Open Source Software},  
  year=2019,
  doi = {10.21105/joss.01667},
  url = {https://doi.org/10.21105/joss.01667}
}
