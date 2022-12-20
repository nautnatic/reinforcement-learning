# Architecture
## Beacon-centered coordinate system
Because the location of the beacon changes each time the beacon is reached or the epoch ends, the states need to be converted in a beacon-centred coordinate system.

## States
The beacon location always needs to be a separate state with one pixel, so the target position can be reached
accurately.
The other states should be areas to minimize the number of states and decrease learning duration.
Exactly above/below or left/right the beacon, the beacon only needs to move in one direction.
Therefore, a line with the width of one pixel should be used.

The other areas should be quadratic.

For neural networks the states are not set to pixel areas to reduce complexity. Instead, each pixel is used as a separate state.