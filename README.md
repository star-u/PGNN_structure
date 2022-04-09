# PGNN_structure
Usage For Running With Python
Packages
Cuda Home To Export

* export CUDA_HOME=/new_disk_B/dyx/anaconda3/pkgs/cudatoolkit-10.1.243-h6bb024c_0/

* export CUDA_HOME = YourAnacondaPath/anaconda3/pkgs/cudatoolkit-your_vesion_to_replace
Eval

* python eval.py --cfg experiments/vgg16_pgnn_structure_willow.yaml --epoch 6
Train

* python train_eval.py --cfg experiments/vgg16_pgnn_structure_willow.yaml ​ python train_eval.py --cfg experiments/vgg16_pgnn_structure_voc.yaml ​ python train_eval.py --cfg experiments/vgg16_pgnn_structure_cub.yaml ​ python train_eval.py --cfg experiments/vgg16_pgnn_structure_imcpt.yaml
Dataset

* Here we release four datasets in our experiments. More details see in the "data/dataset/download.txt". The sintel dataset will be implemented in the future.
Device

* If you want to change device number on GPU, please modify "Motif_Position/utils_pgnn.py" 's device number.
Iteration

* Use iteration=True in "PGNN_structure/model.py" to see how iterative position and structure GCN works.
Cite

* If you want to use our implementations, please cite as follow:
