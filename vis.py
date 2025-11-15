import matplotlib.pyplot as plt
import numpy as np

timestamps = np.array([
    6772257, 6772470, 6772734, 6773237, 6773341,
    6777410, 6777607, 6777791, 6778350, 6778448,
    6782498, 6782727, 6782907, 6783495, 6783602,
    6787629, 6787845, 6788070, 6788599, 6788706,
    6792745, 6792936, 6793133, 6793736, 6793838,
    6797826, 6798028, 6798242, 6798824, 6798943,
    6802855, 6803106, 6803383, 6803943, 6804066
])

# Convert timestamps to relative time (microseconds → milliseconds)
t_rel = (timestamps - timestamps[0]) / 1000.0  # ms

plt.figure(figsize=(10, 4))
plt.plot(t_rel, marker="o")
plt.title("Event Timestamps at Pixel vs. Event Index")
plt.xlabel("Event Index")
plt.ylabel("Time (ms)")
plt.grid(True)
plt.show()
