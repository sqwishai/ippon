
const uint N = sizes[0];  // at run time, not a template constant: one compiled kernel for every number of rows
uint i = thread_position_in_grid.x;
if (i >= N) { return; }
uint row = i / WIDTH;
uint col = i - row * WIDTH;
float g = static_cast<float>(gu[row * 2 * WIDTH + col]);
float u = static_cast<float>(gu[row * 2 * WIDTH + WIDTH + col]);
ACTIVATION
out[i] = static_cast<T>(a * u);
