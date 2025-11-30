# ECE9611_Project
A project on ASL translating

Training.py is the main file implementation of all models

Original Video Data for .mp4 is 16G, flushed_data_new for .npz is 120m, which both are impossible to upload. Run VideoCapture2.py could download those file.

flush_skeleton_dataset.py is the one normalizing .npz

training.py is the one of main training process. There are built in models classes of 1DCNN, GNU, MLP 2DCNN and Transformers. Cancle comment out marks in main to run them. 

helper_data.py is one small helper that help clean out abnormal data (but it didn't help. It's a .npz level fix, so the next step might be re-extract features from .mp4, but it would really take time)

plot.py takes "--run ./plot_data/xxx" as arguement to use data given after training to plot
