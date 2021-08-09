import numpy as np
import matplotlib.pyplot as plt

# Define sample labels

classes = ['Car', 'Duck', 'Face', 'Motorbike', 'Winebottle']
# IPCA+P+S
confusion_mat_1 = np.array(
    [[0.9450, 0.5900, 0.9709, 0.4920, 0.6250], [0.6140, 0.8970, 0.9518, 0.4460, 0.6490],
     [0.6270, 0.6110, 1.000, 0.4890, 0.6590], [0.590, 0.5430, 0.989, 0.7990, 0.627],
     [0.6730, 0.5010, 0.9588, 0.4720, 0.9430]])
# PCA+P+S
confusion_mat_2 = np.array(
    [[0.898, 0.678, 1, 0.515, 0.833], [0.744, 0.852, 1, 0.482, 0.822], [0.709, 0.608, 1, 0.496, 0.607],
     [0.73, 0.713, 1, 0.801, 0.827], [0.75, 0.615, 1, 0.502, 0.928]])
# PCA
confusion_mat_3 = np.array(
    [[0.832, 0.512, 0.706, 0.426, 0.425], [0.591, 0.864, 0.795, 0.365, 0.498], [0.482, 0.424, 1, 0.358, 0.328],
     [0.383, 0.259, 0.75, 0.69, 0.283], [0.543, 0.606, 0.766, 0.370, 0.877]])
# IPCA
confusion_mat_4 = np.array(
    [[0.947, 0.608, 0.949, 0.509, 0.538], [0.552, 0.902, 0.908, 0.417, 0.52], [0.482, 0.424, 1, 0.358, 0.328],
     [0.594, 0.549, 0.9, 0.792, 0.425], [0.64, 0.678, 0.96, 0.476, 0.92]])

confusion_mat_5 = np.array([[0.9450, 0.5900, 0.9709, 0.4920, 0.6250, 0.898, 0.678, 1, 0.515, 0.833, 0.832, 0.512,
                             0.706, 0.426, 0.425, 0.947, 0.608, 0.949, 0.509, 0.538],
                            [0.6140, 0.8970, 0.9518, 0.4460, 0.6490, 0.744, 0.852, 1, 0.482, 0.822, 0.591, 0.864,
                             0.795, 0.365, 0.498, 0.552, 0.902, 0.908, 0.417, 0.52],
                            [0.6270, 0.6110, 1.000, 0.4890, 0.6590, 0.709, 0.608, 1, 0.496, 0.607, 0.482, 0.424, 1,
                             0.358, 0.328, 0.482, 0.424, 1, 0.358, 0.328],
                            [0.590, 0.5430, 0.989, 0.7990, 0.627, 0.73, 0.713, 1, 0.801, 0.827, 0.383, 0.259, 0.75,
                             0.69, 0.283, 0.594, 0.549, 0.9, 0.792, 0.425],
                            [0.6730, 0.5010, 0.9588, 0.4720, 0.9430, 0.75, 0.615, 1, 0.502, 0.928, 0.543, 0.606,
                             0.766, 0.370, 0.877, 0.64, 0.678, 0.96, 0.476, 0.92]])
# ind1 = np.argpartition(confusion_mat_5[4], -4)[-16:]
# print(ind1[:4])
# print(confusion_mat_5[0][ind1])
confusion_mat_6 = np.array([[0.75, 0.5, 1, 0, 0.25], [0.5, 0.75, 1, 0, 0.5], [0.75, 0.75, 1, 0, 0.75], [0.25, 0, 1, 0.75, 0.5], [0, 0.5, 1, 0, 1]])
confusion_mat_7 = np.array([[0.75, 0.25, 1, 0.5, 0.75], [0.5, 0.75, 1, 0, 0.75], [0.75, 0.5, 1, 0.5, 0.5], [0.5, 0.5, 1, 0.75, 1], [0.25, 0.25, 1, 0.5, 0.75]])
confusion_mat_8 = np.array([[0.5, 0.5, 0.75, 0, 0.25], [0.25, 0.75, 0.5, 0.25, 0.25], [0, 0.5, 1, 0.25, 0.25], [0.25, 0.25, 0.75, 0.5, 0], [0.5, 0.5, 0.75, 0.75, 0]])
confusion_mat_9 = np.array([[1, 0.25, 1, 0, 0], [0.25, 1, 1, 0, 0], [0.25, 0.25, 1, 0, 0], [0.75, 0, 1, 0.25, 0], [0.25, 0.25, 1, 0, 0.75]])
plt.imshow(confusion_mat_9, cmap='Oranges')

plt.colorbar()
xlocations = np.array(range(len(classes)))
plt.xticks(xlocations, classes)
plt.yticks(xlocations, classes)
plt.tick_params(bottom=False, top=False, left=False, right=False)
plt.ylabel('Training Category')
plt.xlabel('Testing Category')
#plt.title('PCA-GM+Position+Structure Accuracy (diag:0.8958, all:0.7644)')
#plt.title('IPCA-GM Accuracy (diag:0.9122, all:0.6550)')
#plt.title('PCA-GM+Position+Structure Rankings')
plt.title('IPCA-GM Accuracy Rankings')
plt.grid(which='minor')
for first_index in range(len(confusion_mat_4)):
    for second_index in range(len(confusion_mat_4[first_index])):
        plt.text(first_index, second_index, confusion_mat_4[second_index][first_index], horizontalalignment='center')
plt.show()

# print(np.sum(confusion_mat_4) / 25)  # 70.65 76.44 56.53 65.50
