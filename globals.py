# chessboard size
m_chess = 9
n_chess = 6

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension
N = 7 # N is the number of lane-line data that will be pushed to the stack to avarage some measurments to smooth the detected lane area