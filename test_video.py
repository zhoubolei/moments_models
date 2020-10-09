"""Test pre-trained RGB model on a single video.

Authors: Bolei Zhou, Alex Andonian and Mathew Monfort

This script accepts an mp4 video as the command line argument --video_file.

Alternatively, one may instead provide the path to a directory containing
video frames saved as jpgs, which are sorted and forwarded through the model.

Optionally, one can generate a new video --rendered_output from the frames
used to make the prediction with the predicted category in the top-left corner.

"""

import os
import argparse
import moviepy.editor as mpy

import torch.optim
import torch.nn.parallel
from torch.nn import functional as F

import models
from utils import extract_frames, load_frames, render_frames


# options
parser = argparse.ArgumentParser(description="test on a single video")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--multi', dest='multi', action='store_true')
parser.add_argument('--rendered_output', type=str, default=None)
parser.add_argument('--num_segments', type=int, default=16)
parser.add_argument('--arch', type=str, default='resnet3d50', choices=['resnet50', 'resnet3d50'])
args = parser.parse_args()

# Load model
if args.multi:
    args.arch = 'multi_resnet3d50'

model = models.load_model(args.arch)

# Get dataset categories
if args.multi:
    categories = models.load_categories('category_multi_momentsv2.txt')
else:
    categories = models.load_categories('category_momentsv2.txt')

# Load the video frame transform
transform = models.load_transform()

# Obtain video frames
if args.frame_folder is not None:
    print('Loading frames in {}'.format(args.frame_folder))
    import glob
    # here make sure after sorting the frame paths have the correct temporal order
    frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
    frames = load_frames(frame_paths)
else:
    print('Extracting frames using ffmpeg...')
    frames = extract_frames(args.video_file, args.num_segments)


# Prepare input tensor
if 'resnet3d50' in args.arch:
    # [1, num_frames, 3, 224, 224]
    input = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
else:
    # [num_frames, 3, 224, 224]
    input = torch.stack([transform(frame) for frame in frames])

# Make video prediction
with torch.no_grad():
    logits = model(input)
    h_x = F.softmax(logits, 1).mean(dim=0)
    probs, idx = h_x.sort(0, True)

# Output the prediction.
video_name = args.frame_folder if args.frame_folder is not None else args.video_file
print('RESULT ON ' + video_name)
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))

# Render output frames with prediction text.
if args.rendered_output is not None:
    prediction = categories[idx[0]]
    rendered_frames = render_frames(frames, prediction)
    clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
    clip.write_videofile(args.rendered_output)
