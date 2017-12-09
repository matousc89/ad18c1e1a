import numpy as np
import matplotlib.pylab as plt
import signalz
import padasip as pa

def SNR(x, v):
    # get standard deviation of signal
    if hasattr(x, '__len__') and (not isinstance(x, str)):
        s1 = np.std(d)**2
    else:
        s1 = x**2
    # get standard deviation of noise
    if hasattr(x, '__len__') and (not isinstance(x, str)):
        s2 = np.std(v)**2
    else:
        s2 = v**2
    return 10*np.log10(s1/s2)

def roc_curve(predicted_values, actual_conditions, steps=100, interpolation_steps=100):
    # convert to boolean array
    actual_conditions = actual_conditions != 0
    # get maximum and minimum
    predicted_max = predicted_values.max()
    predicted_min = predicted_values.min()
    # range of criteria
    crits = np.linspace(predicted_min, predicted_max, steps)
    # empty variables
    tp = np.zeros(steps)
    fp = np.zeros(steps)
    tn = np.zeros(steps)
    fn = np.zeros(steps)
    for idx, crit in enumerate(crits):
        # count stuff
        tp[idx] = ((predicted_values > crit) * actual_conditions).sum()
        fn[idx] = ((predicted_values <= crit) * actual_conditions).sum()
        fp[idx] = ((predicted_values > crit) * np.invert(actual_conditions)).sum()
        tn[idx] = ((predicted_values <= crit) * np.invert(actual_conditions)).sum()
    # calculations
    total = tp + fp + tn + fn
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = (tp + tn) / total
    # AUROC integration
    points_x = np.linspace(0, 1, interpolation_steps)
    points_y = np.interp(points_x, (1-spe)[::-1], (sen)[::-1])
    auroc = np.sum(points_y*(1/interpolation_steps))
    return sen, spe, acc, auroc

def sample_entropy(x, m=2, r=0, distance="chebyshev"):
    # select r if it is not provided
    if r == 0:
        r = 0.3*np.std(x)
    # create embeded matrix
    xa = pa.preprocess.input_from_history(x, m+1)
    xb = pa.preprocess.input_from_history(x, m)[:-1]
    N = len(xa)
    A = np.zeros(N, dtype="float")
    B = np.zeros(N, dtype="float")
    # iterate over all samples
    for i in range(N):
        xia = xa[i]
        xib = xb[i]
        if distance == "chebyshev":
            da = np.max(xia-xa, axis=1)
            db = np.max(xib-xb, axis=1)
            crit = r
        elif distance == "euclidean":
            da = np.sum((xia-xa)**2,axis=1)
            db = np.sum((xib-xb)**2,axis=1)
            crit = r**2
        A[i] = np.sum(da < crit)
        B[i] = np.sum(db < crit)
    # estimate the output and insert zero padding
    out = np.zeros(len(x))
    out[m:] = -np.log10(A/B)
    return out


## inputs
np.random.seed(101)
change_number = 200 # 200
change_samples = 1000
positive_samples = 500
skip_on_start = 20000 #20000
n = 10
system = 1
skip_tests = 10


## precalculated stuff an data generating
total_len = change_number * change_samples

# inputs
x = np.random.normal(0, 1, (total_len, n))

# parameters
h = np.random.normal(0, 1, (change_number, n))
h = np.repeat(h, change_samples, axis=0)

# output
v = np.random.normal(0, 1., total_len)
d = np.sum(x * h, axis=1) + v

## plot of changes in data changes
# for k in range(change_number):
#     plt.axvline(k*change_samples, color='k', linestyle='--')
# plt.plot(d[:])
# plt.xlim(0, 10000)
# plt.tight_layout()
# plt.show()


## ADAPTIVE FILTER STUFF (for LE and ELBND)
f = pa.filters.FilterNLMS(n=n, mu=1.5, w="zeros")
y, e, w = f.run(d, x)
dw = np.zeros(w.shape)
dw[0:-1] = np.abs(np.diff(w, axis=0))
e = abs(e)
dw = abs(dw)

# get ELBND
elbnd = pa.detection.ELBND(w, e, function="sum")

# get LE
le = pa.detection.learning_entropy(w, m=1000, order=1, alpha=[6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.])

## SAMPLE ENTROPY STUFF - too slow to be enabled all the time

# get SE
# se = sample_entropy(d, m=2)

# FUZZY SYSTEMS
# ....



## OBSOLETE STUFF USEFULL IN FUTURE
# # get correlations
# for idx in range(dw.shape[1]):
#     print(np.corrcoef(abs(e[skip_on_start:]), abs(dw[skip_on_start:,idx]))[0,1])

# ## normalization not used this time
# print(np.mean(e), np.std(e))
# print(np.mean(dw), np.std(dw))
# print(np.mean(elbnd), np.std(elbnd))
# norma = 3
# #e = e / np.mean(e) / norma
# dw = dw / np.mean(dw) / norma
# elbnd = elbnd / np.mean(elbnd) / norma
# k = 1
# x0 = 6
# #e = 1 / (1 + np.exp(-k*(e-x0)))
# dw = 1 / (1 + np.exp(-k*(dw-x0)))
# elbnd = 1 / (1 + np.exp(-k*(elbnd-x0)))

## DATA FOR REPRESENTATION
methods = [
    {"name": "LE", "data": le, "line": "b"},
    {"name": "ELBND", "data": elbnd, "line": "g"},
    {"name": "ERR", "data": e, "line": "k"}
    # {"name": "SE", "data": se, "line": "r"}, # will be enabled in final test
]
for method in methods:
    method["reduced"] = np.zeros(change_number*2)


# plot ROC for all active methods
actual_values = signalz.steps(1, [1,0], repeat=change_number)
for idx in range(change_number):
    start = idx * change_samples
    end = (idx+1) * change_samples
    for method in methods:
        method["reduced"][(idx*2)] = method["data"][start:start+positive_samples].max()
        method["reduced"][(idx*2)+1] = method["data"][start+positive_samples:end].max()
for method in methods:
    method["sen"], method["spe"], method["acc"], method["auroc"] = roc_curve(method["reduced"][skip_tests:], actual_values[skip_tests:], steps=100)
    plt.plot(1-method["spe"], method["sen"], method["line"])
    print(method["name"], "\t", method["acc"].max(), "\t", method["auroc"])
plt.show()



# some usefull stuff for debuging in plots
# actual_rule = np.zeros(change_samples)
# actual_rule[:positive_samples] = 1
# actual_rule = signalz.steps(1, actual_rule, repeat=change_number)

# ax1 = plt.subplot(511)
# plt.plot(d[skip_on_start:])
# plt.plot(y[skip_on_start:])
#
# # plt.subplot(512, sharex=ax1)
#
#
#
# # plt.subplot(513, sharex=ax1)
# # plt.plot(se[skip_on_start:])
# # plt.plot(actual_rule[skip_on_start:])
# # plt.title("SE")
#
# plt.subplot(514, sharex=ax1)
# plt.plot(elbnd[skip_on_start:])
# plt.title("ELBND")
# # plt.plot(actual_rule[skip_on_start:])
#
# plt.subplot(515, sharex=ax1)
# plt.plot(le[skip_on_start:])
# plt.title("LE")
#
# plt.tight_layout()
# plt.show()