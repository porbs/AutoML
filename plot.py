import matplotlib.pyplot as plt
import numpy as np
import json

filename = 'small_gen_39.json'

with open(filename) as fp:
    data = json.load(fp)

x = []
mins = []
maxs = []
means = []

for i, generation in enumerate(data):
    x.append(i)
    accs = np.array(list(map(lambda val: val['accuracy'], list(
        filter(lambda val: val['accuracy'] > 0, generation)))))
    mins.append(np.min(accs))
    maxs.append(np.max(accs))
    means.append(np.mean(accs))

plt.fill_between(x, mins, maxs)
plt.plot(x, means, color='black')
plt.show()
