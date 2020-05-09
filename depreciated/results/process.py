import pickle
import numpy as np
import matplotlib.pyplot as plt


latent_data = pickle.load(open("latent.pkl", "rb"))
baseline_data = pickle.load(open("baseline.pkl", "rb"))

L1 = np.zeros((len(latent_data), 1))
L2 = np.zeros((len(baseline_data), 1))
Z = np.zeros((40, 100))
for index, item in enumerate(latent_data):
    L1[index,:] = item[0]
    Z[:, index] = np.reshape(item[1], 40)
for index, item in enumerate(baseline_data):
    L2[index,:] = item[0]

L1m = np.mean(L1)
L1std = np.std(L1)
print(L1m, L1std)

L2m = np.mean(L2)
L2std = np.std(L2)
print(L2m, L2std)

plt.plot(L1)
plt.plot(L2)
plt.show()

Zm = np.mean(Z, axis=1)
plt.plot(Zm)
plt.show()
