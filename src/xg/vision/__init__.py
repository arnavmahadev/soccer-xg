"""Photo / video -> 2D pitch extrapolation.

Detects players and the ball in a broadcast or sideline image, assigns each
player to one of two teams by jersey colour, and projects every detection onto
the StatsBomb-style 120x80 pitch via a user-supplied homography. The same
per-frame routine drives both single photos and scrubbed video.
"""
