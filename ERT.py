from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import numpy as np

# Coleta de Dataset

dataset = []
dataset_name = 'iris'
with open(dataset_name + '.data') as f:
		conteudo = f.readlines()
for x in conteudo:
	dataset.append(x.strip().split(','))

dataset = np.asarray(dataset).astype(np.float)

target = []

for x in dataset:
	target.append(x[4])
target = np.asarray(target).astype(np.int)

dataset =  np.delete(dataset, -1, axis=1)

# Dateset definido como as 4 colunas de atributos do arquivo
# Target definido como a última coluna do arquivo (Classe)


# Inicializando o RFC

clf = RandomForestClassifier(n_estimators=2)

clf = clf.fit(dataset, target)

fig = plt.figure(1, figsize=(4, 3))

plt.clf()
plt.suptitle(str(clf.n_estimators) + " Estimadores")
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

labels = ['x', 'y', 'z']
ax.scatter(dataset[:, 3], dataset[:, 0], dataset[:, 2])
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.set_zlabel('Atributo 3')
plt.show()