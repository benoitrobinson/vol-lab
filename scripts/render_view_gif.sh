#!/bin/sh
# Turns the recorded cast into the README animation. Kept beside the recorder
# so the theme and the speed are part of the repo rather than shell history.
set -e
cd "$(dirname "$0")/.."
agg \
  --theme 000000,d8dee9,1a1a1a,bf616a,a3be8c,ebcb8b,5e81ac,b48ead,88c0d0,d8dee9,4c566a,bf616a,a3be8c,ebcb8b,81a1c1,b48ead,8fbcbb,eceff4 \
  --font-size 15 --line-height 1.3 --speed 1.35 \
  figures/vol-lab-view.cast figures/vol-lab-view.gif
