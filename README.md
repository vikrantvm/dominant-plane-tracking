# DominantPlaneTracking

This app tracks a dominant planar surface in structured environments.
It uses the idea that planar surfaces are related by homographies. The app
uses SIFT feature descriptors and FLANN based matching. Requires initializing
a region of interest to start tracking.

To run:
cmake .
make
./dom name_of_video_stream
