# ECE9611_Project
A project on ASL translating

Training.py is the main file implementation of all models

Original Video Data for .mp4 is 16G, flushed_data_new for .npz is 120m, which both are impossible to upload. Run VideoCapture2.py could download those file, but it took 20 hours in total for us. We used
MSL_TRAIN25, MSL_TEST25, MSL_VAL25 and MSL_test and mixed them together as my general dataset.

VideoProcess.py do hand detect from .mp4 into .npz, which significantly makes the training into possible. Transforming all .mp4 into .npz took us 8 hours.

flush_skeleton_dataset.py is the one normalizing .npz, it should take no more than half hour.

training.py is the one of main training process. There are built in models classes of 1DCNN, GNU, MLP 2DCNN and Transformers. Cancel comment out marks in main to run them. 

helper_data.py is one small helper that help clean out abnormal data (but it didn't help. It's a .npz level fix, so the next step might be re-extract features from .mp4, but it would really take time)

plot.py takes "--run ./plot_data/xxx" as arguement to use data given after training to plot

The process to run this project should be:

1. VideoCapture2.py
2. VideoProcess.py
3. flush_skeleton_dataset.py
4. training.py

optional(after above):

5. helper_data.py
6. plot.py

Requirements.txt

1. Tensor version had a confilt with medipipe in numpy version in all version, for us just let one collect video then another one train
2. yt-dpl seems required an external exe file and package
