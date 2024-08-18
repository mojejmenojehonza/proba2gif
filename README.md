# Proba -> Gif
Automatically rotates and centers images from proba2 dumps and writes it into .gif or .webp file
## Installation
### Prerequisities
python3, imageio , opencv and numpy
run this to install all

`pip install opencv-python numpy imageio`
### Clone
```
git clone https://github.com/mojejmenojehonza/proba2gif # or manually download zip and extract it
cd proba2gif
python3 main.py -h
```
### Usage
`cli.py [-h] [--output_file OUTPUT_FILE] [--calibration_image CALIBRATION_IMAGE] [--frame_duration FRAME_DURATION] [--tint {normal,green,pink,blue}] [--brightness BRIGHTNESS] [--contrast_factor CONTRAST_FACTOR] input_dir
cli.py: error: the following arguments are required: input_dir`
## Example
`python3 cli.py SWAP --output_file proba2.webp --tint normal`

https://github.com/user-attachments/assets/b2af8fe5-5fd7-4aa5-a399-63d019edea57

`python3 cli.py SWAP --output_file proba2.webp --tint blue`

https://github.com/user-attachments/assets/4a7a6381-294e-448e-8095-82852803bf89

## Credits
credits to muellermilch.de on discord for providing me with a proba2 dump
