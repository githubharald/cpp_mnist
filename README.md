# Simple MNIST classifier in plain C++
A neural network built in ~260 lines of plain C++ code without any dependencies.
The model is trained to distinguish the handwritten digits "0" and "1" from the MNIST dataset.


````
Training model...
Dataset size: 12665
Epoch: 0 Sample: 0 Loss: 2.44357
Epoch: 0 Sample: 1000 Loss: 0.230895
...
Epoch: 9 Sample: 12000 Loss: 0.00112786
Testing model...
Dataset size: 2115
Accuracy: 0.994326





              X
         XXXXXXXXXX
        XXXXXXXXXXXX
        XXX      XXXX
        XX        XXX
        XX         XXX
       XX          XXX
       XX           XX
       XX           XX
      XX            XX
      XX            XX
      XX            XX
      XX           XXX
      XX           XX
      XX          XXX
      XX         XXX
      XX       XXXX
      XXXXXXXXXXXXX
       XXXXXXXXXX
        XXXXXXX




Predicted: 0 (0.155461) Target: 0
Press ENTER to see next sample...






                X
               XXX
               XXX
              XXX
              XXX
             XXXX
             XXX
             XXX
             XXX
            XXX
            XXX
            XXX
           XXX
           XXX
           XXX
           XX
          XXX
          XXX
          XXX
           X



Predicted: 1 (0.962643) Target: 1
Press ENTER to see next sample...
````

# How to run it
* Unzip the file mnist_data.zip, make sure the files are in the same folder as the cpp file
* Compile the C++ code, e.g. with `g++ cpp_mnist.cpp` on Linux, or by using Visual Studio on Windows
* Run the program, e.g. by executing `./a.out` on Linux
* The model takes ~10s to train, then evaluates on the testset (should get ~99% accuracy), and then shows samples and their predictions


# Notes
* The model is in fact a regression model (squared error loss and linear activation in final layer) to keep things as simple as possible, and is trained to output 0.0 for "0" and 1.0 for "1", however, raw outputs can also occur outside of that range
* The `RegressionModel` class allows configuring the model (e.g., number of layers)
* Good results are achieved with 2-4 layers and 5-15 unites per hidden layer
* The backpropagation code follows the algorithm outlined in the Deep Learning book from Bishop
* The dataset is created from the original MNIST dataset
