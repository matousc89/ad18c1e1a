import numpy as np
import matplotlib.pylab as plt
import signalz
import padasip as pa

"""
No text
"""


setups = [
    {"drift": "none", "plot_pos": 221, "drift_label": "none"},
    {"drift":"ramp", "plot_pos": 222, "drift_label": "ramp"},
    {"drift":"sinus", "plot_pos": 223, "drift_label": "sinus waves"},
    {"drift":"both", "plot_pos": 224, "drift_label": "ramp + sinus_waves"},
]

setup = setups[0]

np.random.seed(101)

plt.figure(figsize=(15,10))


# system change

## eeg
np.random.seed(101)
change_number = 500 # 500
change_samples = 500 # 500
positive_samples = 50 # 50
neutral_samples = 0
skip_on_start = 20000 #20000
n = 10
system = 1
skip_tests = 10

## precalculated stuff an data generating

total_len = change_number * change_samples
d = np.loadtxt("data_eeg.txt")[:total_len] * 5

# print(np.mean(d), np.std(d))

for idx in range(1, change_number):
    d[idx * change_samples] += np.random.normal(0, 1)
x_d = pa.input_from_history(d, n)[:-1]
d_d = d[n:]
x = np.zeros((total_len, n))
d = np.zeros(total_len)
x[n:] = x_d
d[n:] = d_d


if setup["drift"] in ["ramp", "both"]:
    d += np.linspace(0, 1, total_len)
if setup["drift"] in ["sinus", "both"]:
    d += signalz.sinus(total_len, period=100000, amplitude=1)

# # plt.plot(d[8000:10000], "k")
plt.subplot(212)
plt.plot(d[10000:12000], "k")
plt.axvline(500, linestyle=":", color="k")
plt.axvline(1000, linestyle=":", color="k")
plt.axvline(1500, linestyle=":", color="k")
plt.title("b) Data for outlier detection (Ecgsyn output)")
plt.xlabel("Discrete time index [-]")
plt.ylabel("Simulation data [-]")


## precalculated stuff an data generating
total_len = change_number * change_samples

# inputs
x = np.random.normal(0, 1, (total_len, n))

# parameters
h = np.random.normal(0, 1, (change_number, n))
h = np.repeat(h, change_samples, axis=0)

# output
v = np.random.normal(0, 1, total_len)
d = np.sum(x * h, axis=1) + v

if setup["drift"] in ["ramp", "both"]:
    d += np.linspace(0, 2, total_len)
if setup["drift"] in ["sinus", "both"]:
    d += signalz.sinus(total_len, period=100000, amplitude=4)

# plt.plot(d[8000:10000], "k")
plt.subplot(211)
plt.plot(d[10000:12000], "k")
plt.axvline(500, linestyle=":", color="k")
plt.axvline(1000, linestyle=":", color="k")
plt.axvline(1500, linestyle=":", color="k")
plt.title("a) Data for system change point detection")
plt.xlabel("Discrete time index [-]")
plt.ylabel("Simulation data [-]")

plt.tight_layout()
plt.savefig("figs/fig_show.png")
plt.show()

