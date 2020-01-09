# PriorLearning

### 1. Dependencies
* Numpy
* Matplotlib
* OpenCV for Python
* [HDF5 for Python](http://www.h5py.org/)
* PyTorch

### 2. Reproduce results
To train the models you have two possibilities:
- Run the 'train.py' directly with the desired parameters. The list of possible parameters is printed when the file is run with the '--help' argument. Note the the only two supported optimizers are 'SGD' and 'Adam', and the '--augmentation' argument supports either 'no' data augmentation, 'standard' or 'vae'. We recommend writing a script to automate this process when running on multiple dataset sizes.
- Run the 'train_test_model.sh' script to automatically get the results for the same dataset sizes we used. This scripts supports the following arguments: '-c' which should be set to 'true' to create the h5py dataset; '-s' to set the seed; '-o' to specify the optimizer; '-lr' to specify the learning rate; '-b' to specify the batch size; '-k' to specify the sigma parameter of the gaussian kernel used to blur the images; '-d' to specify the number of layers in the model; '-a' to specify the type of data augmentation. Note that if script displays the 'bad interpreter ...' message you can run this command to fix the problem: `sed -i 's/\r//' train_test_model.sh`.

### 3. Visualize the results
You can use the 'Results' notebook to visualize the results. You need to have the 'logs' folder at the same location as your notebook for it to work. The code to generate all the graphs used in the report is on the notebook and can be used as is as long as you have run the code with the parameters you want to visualize.

If you want to run all the experiments you can simply run the 'experiments.sh' script but the running time will be very long!
