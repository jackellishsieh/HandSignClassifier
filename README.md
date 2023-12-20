# HandSignClassifier
 A neural network built in Java from scratch for recognizing numerical hand signs

## Overview
The repository consists of three components:
1. **ABCDImageNetwork**: A four-layer feedforward neural network implemented via backpropagation with configurable settings:
   1. Training/running mode
   2. Weight randomization/file input
   3. Weight file output
   4. Data file input
   5. Network layer sizes
   6. Training parameters: $\lambda$, $E_{max}$, and $n_{iterations}$

2. **imageProcessing**: Converted data files for hand sign images.

3. **handImages**: Raw numerical hand sign images.

This project was created for ATCS: Neural Nets at The Harker School, taught by Dr. Nelson.

## Usage

### Dependencies
Standard Java libraries.

### Running
1. Edit the control file, which has four arguments:
   1. `doTrainNotRun`: Whether to train or run (boolean)
   2. `networkConfigurationFilename`: The network configuration filename (string)
      - This network configuration file contains values for network parameters.
   4. `inputSetFilename`: The input set filename to run the network on (string)
      - This input set file in turn contains an ordered list of individual input member files. 
   5. `targetSetFilename`: The target set filename to compare with the network's outputs (string)
      - This output set file in turn contains an ordered list of individual target member files.
3. Run _ABCDNetworkTester.java_ from the terminal with the control filename as the first argument.
4. The program will create and run the network based on the provided settings.
6. Once finished training/running, the program will print the network specifications and a comparison table of network outputs and target outputs to the console.
   1. If specified, the network will also save weights to the specified weight output file.

### Help
See the repository's control, network configuration, input set, and target set files for examples.
