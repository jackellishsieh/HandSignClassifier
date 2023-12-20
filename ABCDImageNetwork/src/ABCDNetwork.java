/*
 * A-B-C-D network class that uses backpropagation and file IO.
 * 
 * Author: Jack Hsieh
 * Date of creation: November 11, 2021
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import java.util.Scanner;

/*
 * Defines an A-B-C-D multilayer perceptron that can run and train on inputs using gradient descent via backpropagation.
 * 
 * The number of input, hidden, and output units are variable. The number of hidden layers is fixed to 2.
 * Weights are randomized or read from a file. 
 * File IO is used to construct the network, read and write weights, and (if desired) train and run on a set of members.
 * The network can be configured to forgo training capabilities if the network will only run.
 */
public class ABCDNetwork
{
   /*
    * Total number of layers, including input, hidden, and output layers
    */
   private int NUM_LAYERS = 4;         // Not user configurable!
   
   /*
    * Number of units in each layer
    * LAYER_SIZES[0] is the number of input units
    * LAYER_SIZES[NUM_LAYERS - 1] is the number of output units
    */
   private int[] LAYER_SIZES;          // Of size NUM_LAYERS
   
   /*
    * Units
    */
   private double[][] a;               // Each row is a layer: row 0 is the input layer, row 1 is the first hidden layer, etc.
   
   /*
    * Weights
    */
   private double[][][] w;
   
   /*
    * Weight initialization
    */
   private boolean randomizeWeights;   // If true, randomize the weights. Otherwise, load from a file

   private double RANDOM_WEIGHT_MIN;   // Only used if randomizeWeights is true
   private double RANDOM_WEIGHT_MAX;   // Only used if randomizeWeights is true
   
   private File weightsInputFile;      // Only used if randomizeWeights is false
   
   /* 
    * Training parameters
    */
   private boolean allocateForTraining;
   private double lambda;
   private double errorThreshold;
   private int maxIterations;
   
   /*
    * Targets
    */
   private double[] T;                 // Only allocated if allocateForTraining is true
   
   /*
    * Training hidden-output details. Only allocate if allocateForTraining is true.
    * Both arrays contain NUM_LAYER rows
    */
   private double[][] Theta;           // The first and last (the input and output layers) are null since they are never used
   private double[][] Psi;             // The first two layers (the input and leftmost hidden layer) is null since they are never used
   
   /*
    * Training flags
    */
   private boolean maxIterationsReached;
   private boolean errorThresholdSatisified;

   /*
    * Training performance variables
    */
   private int numIterations;
   private double maximumSetError;
   
   /*
    * Weight output configuration
    */
   private File weightsOutputFile;
   private int saveWeightsEvery;
   private boolean saveWeightsAtEnd;
   
   /*
    * Loop indices
    */
   
   private int alpha;                  // Layer index
   
   /*
    * Loop indices for specific layers
    * When each layer needs to be treated differently
    */
   private int i;                      // Output layer
   private int j;                      // Hidden layer 2
   private int k;                      // Hidden layer 1
   private int m;                      // Input layer
     
   /*
    * Generalized loop indices
    * When each layer is treated the same way
    */
   private int beta;                   // Current layer
   private int gamma;                  // Layer to the right of the current layer
   
   /*
    * Prints a message every intervals with the error and milliseconds
    */
   private int printEvery = 20;
   
   /*
    * Constructs the network using a network configuration file
    * 
    * parameters: networkConfigFile is the file path object identifying the network configuration file
    * postconditions: If the given file path object identifies a valid control file, construct the network according to the configuration file.
    *                 Allocates and sets network sizes, weights (via randomization or from a file), training details if necessary, 
    *                 and the weights output configuration.
    *                 If the given file path object does not identify an existing file, abort construction and throw a FileNotFoundException.
    *                 If the given file path object identifies an improperly formatted control file, abort construction and throw an IllegalArgumentException.
    */
   public ABCDNetwork(File networkConfigFile) throws FileNotFoundException, IllegalArgumentException
   {
      Scanner networkConfigFileReader = null;                           // Scanner for reading the networkConstructionFile
      
      /*
       * Check that the network configuration file path object identifies an existing file and open a scanner
       */
      try
      {
         networkConfigFileReader = new Scanner(networkConfigFile);
      } // try
      
      catch (FileNotFoundException fileNotFoundException)               // If the file is not found, label and throw an exception
      {
         throw (new FileNotFoundException("Network configuration file not found: " + fileNotFoundException.getMessage()));
      } // catch (FileNotFoundException fileNotFoundException) 
      
      /*
       * Use the scanner to read the necessary network configuration information
       * Split into four stages: network foundation (sizes), weight initialization, training configuration, and weight output configuration
       */
      try
      {
         this.loadNetworkFoundation(networkConfigFileReader);           // Read in network sizes, allocate units, and allocate weights
         this.loadWeightsInitialization(networkConfigFileReader);       // Read and decide whether to randomize weights or read from a file
         this.loadTrainingConfiguration(networkConfigFileReader);       // Read in training parameters and allocate training arrays if called for
         this.loadWeightsOutputConfiguration(networkConfigFileReader);  // Read in where and how often to save weights
      } // try
      
      catch (Exception exception)                                       // Rethrows exceptions from sub-methods
      {
         throw (exception);
      } // catch (Exception exception)
      
      finally
      {
         networkConfigFileReader.close();                               // Always close the scanner
      } // finally
      
      return;
   } // public ABCDNetworkBP(File networkConfigFile) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Reads the network sizes and allocates the corresponding unit and weight arrays
    * Does not close the scanner
    * 
    * parameters: networkFoundationReader, the scanner to read the network sizes
    * preconditions: networkFoundationReader is initialized to a scanner 
    * postconditions: If the scanner's next (1 + NUM_LAYERS) tokens consist of a valid network configuration file's network sizes,
    *                 load the file's network sizes into the network and allocate unit and weight arrays accordingly.
    *                 The scanner's position therefore advances (1 + NUM_LAYERS) tokens forward.
    *                 Otherwise, throw an IllegalArgumentException.
    *                 The scanner is not closed in this function. 
    */
   private void loadNetworkFoundation(Scanner networkFoundationReader) throws IllegalArgumentException
   {
      /*
       * Read in the information from the file. All read information here is stored by the network permanently.
       */
      try
      {
         networkFoundationReader.useDelimiter(":|\\n|-");               // Use the colon, newline, and hyphen as delimiters
         
         this.LAYER_SIZES = new int[this.NUM_LAYERS];                   // Allocates the layer sizes
         
         networkFoundationReader.next();                                // Read the label "LAYER_SIZES"
         
         /*
          * Read in the layer sizes
          */
         for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
         {
            this.LAYER_SIZES[alpha] = networkFoundationReader.nextInt();
         } // for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
         
         /*
          * Allocate the units and weights using the network sizes
          */
         this.allocateUnits();                                          
         this.allocateWeights();
      } // try
      
      catch (InputMismatchException inputMismatchException)             // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)             // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      return;
   } // private void loadNetworkFoundation(Scanner networkFoundationReader) throws IllegalArgumentException
   
   /*
    * Reads and executes the specified weight initialization (randomization or loading from file) method
    * Does not close the scanner
    * 
    * parameters: weightsInitReader, the scanner to read the weight initialization method and corresponding details
    * preconditions: weightsInitReader is initialized to a scanner
    * postconditions: If the scanner's next eight tokens consist of a valid network configuration file's weight initialization configuration,
    *                 load the file's weight initialization configuration into the network and initialize the weights appropriately
    *                 through either randomization or by loading from a file as indicated.
    *                 The scanner's position therefore advances eight tokens forward.
    *                 If the weights are specified to be loaded from a file and the file is not found, throw a FileNotFoundException
    *                 In all other cases,, throw an IllegalArgumentException.
    *                 The scanner is not closed in this function. 
    */
   private void loadWeightsInitialization(Scanner weightsInitReader) throws FileNotFoundException, IllegalArgumentException
   {
      /*
       * Read in the information from the file. 
       * Not all read information is stored by the network permanently: the method of weights initialization and the weights input filename are stored locally.
       */
      try
      {
         weightsInitReader.useDelimiter(":|\\n");                       // Use the colon and newline as delimiters
         
         weightsInitReader.next();                                      // Read the label "randomizeWeights"
         randomizeWeights = weightsInitReader.nextBoolean();            // Read whether to initialize the weights via randomization. Only used locally.
         
         weightsInitReader.next();                                      // Read the label "RANDOM_WEIGHT_MIN"
         this.RANDOM_WEIGHT_MIN = weightsInitReader.nextDouble();       // Read the inclusive minimum for weight randomization
         
         weightsInitReader.next();                                      // Read the label "RANDOM_WEIGHT_MAX"
         this.RANDOM_WEIGHT_MAX = weightsInitReader.nextDouble();       // Read the inclusive minimum for weight randomization
         
         weightsInitReader.next();                                      // Read the label "weightsInputFilename"
         String weightsInputFilename = weightsInitReader.next();        // Reads the weights input filename. Only used locally
         this.weightsInputFile = new File(weightsInputFilename);        // Save the weight input file as a file path object
      } // try
      
      catch (InputMismatchException inputMismatchException)             // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)  
      
      catch (NoSuchElementException noSuchElementException)             // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      /*
       * If the weights should be initialized via randomization, randomize the weights 
       */
      if (this.randomizeWeights)
      {
         this.randomizeWeights();
      } // if (randomizeWeights)
      
      /*
       * Otherwise, attempt to read weights from a file
       */
      else
      {
         try
         {
            this.loadWeights(this.weightsInputFile);
         } // try
         
         catch (Exception exception)                                    // Rethrows exceptions from sub-methods
         {
            throw (exception);
         } // catch (Exception exception)
      } // if (randomizeWeights)... else
      
      return;
   } // private void loadWeightsInitialization(Scanner weightsInitReader) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Reads the specified training configuration (whether to allocate training details and training parameters)
    * Does not close the scanner
    * 
    * parameters: trainingConfigReader, the scanner to read the training configuration
    * preconditions: trainingConfigReader is initialized to a scanner
    * postconditions: If the scanner's next eight tokens consist of a valid network configuration file's training configuration 
    *                 (lambda, errorThreshold, maxIterations), load the file's training configuration into the network 
    *                 and allocate training arrays if indicated.
    *                 The scanner's position therefore advances eight tokens forward.
    *                 Otherwise, throw an IllegalArgumentException.
    *                 The scanner is not closed in this function. 
    */
   private void loadTrainingConfiguration(Scanner trainingConfigReader) throws IllegalArgumentException
   {
      /*
       * Read in the information from the file. All read information here is stored by the network permanently.
       */
      try
      {
         trainingConfigReader.useDelimiter(":|\\n");                       // Use the colon and newline as delimiters
         
         trainingConfigReader.next();                                      // Read the label "allocateForTraining"
         this.allocateForTraining = trainingConfigReader.nextBoolean();    // Read whether to allocate for training or not
         
         trainingConfigReader.next();                                      // Read the label "lambda"
         this.lambda = trainingConfigReader.nextDouble();                  // Read the adaptive rate
         
         trainingConfigReader.next();                                      // Read the label "errorThreshold"
         this.errorThreshold = trainingConfigReader.nextDouble();          // Read the error threshold during training
         
         trainingConfigReader.next();                                      // Read the label "maxIterations"
         this.maxIterations = trainingConfigReader.nextInt();              // Read the error threshold during training
         
         /*
          * Allocate the training arrays (Thetas, targets, Psis) if desired
          */
         if (this.allocateForTraining)
         {
            this.allocateThetas();
            this.allocateTargets();
            this.allocatePsis();
         } // if (this.allocateForTraining)
      } // try
      
      catch (InputMismatchException inputMismatchException)                // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)  
      
      catch (NoSuchElementException noSuchElementException)                // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      return;
   } // private void loadTrainingConfiguration(Scanner trainingConfigReader) throws IllegalArgumentException
   
   /*
    * Reads the specified weight output configuration (the destination for weight-saving and how often to save weights)
    * Does not close the scanner
    * 
    * parameters: weightsOutputConfigReader, the scanner to read the weights output configuration
    * preconditions: weightsOutputConfigreader is initialized to a scanner
    * postconditions: If the scanner's next six tokens consist of a valid network configuration file's weight output configuration 
    * ```             (weights output file, weight-saving frequency, whether to save at the end),
    *                 load the file's training configuration into the network and allocate training arrays if indicated
    *                 The scanner's position therefore advances six tokens forward.
    *                 Otherwise, throw an IllegalArgumentException.
    *                 The scanner is not closed in this function. 
    */
   private void loadWeightsOutputConfiguration(Scanner weightsOutputConfigReader) throws IllegalArgumentException
   { 
      /*
       * Read in the destination for saving weights and how often to save
       */
      try
      {
         weightsOutputConfigReader.next();                                       // Read the label "weightsOutputFilename"
         this.weightsOutputFile = new File(weightsOutputConfigReader.next());    // Set the weights output file to the given file name
         
         weightsOutputConfigReader.next();                                       // Read the label "saveWeightsEvery"
         this.saveWeightsEvery = weightsOutputConfigReader.nextInt();            // Set the frequency of weight-saving
         
         weightsOutputConfigReader.next();                                       // Read the label "saveWeightsAtEnd"
         this.saveWeightsAtEnd = weightsOutputConfigReader.nextBoolean();        // Set whether to set the weights at the end
      } // try
      
      catch (InputMismatchException inputMismatchException)                      // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)  
      
      catch (NoSuchElementException noSuchElementException)                      // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid network configuration file. The network configuration file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)      
      
      return;
   } // private void loadWeightsOutputConfiguration(Scanner weightsOutputConfigReader) throws IllegalArgumentException
   
   /*
    * Loads the weights from a file with a file confirmation stage
    * 
    * parameters: weightsInputFile is the weights input file path object
    * preconditions: the weight arrays are allocated appropriately
    * postconditions: If the file path object identifies a valid weight file, read the file's provided weights into the network's weights
    *                 If the given file path object does not identify an existing file, abort and throw a FileNotFoundException.
    *                 If the given file path object identifies an improperly formatted weight file, abort and throw an IllegalArgumentException.
    */
   public void loadWeights(File weightsInputFile) throws FileNotFoundException, IllegalArgumentException
   {
      Scanner weightsInputReader;
      
      /*
       * Open the scanner and verify the file path object identifies an existing file
       */
      try
      {
         weightsInputReader = new Scanner(weightsInputFile);   // New scanner to read the weight file
      } // try
      
      catch (FileNotFoundException fileNotFoundException)      // If the file is not found, label and throw an exception
      {
         throw (new FileNotFoundException("Weights input file not found: " + fileNotFoundException.getMessage()));
      } //  catch (FileNotFoundException fileNotFoundException)   
      
      /*
       * Confirm the weight file is a match by comparing network sizes
       * If the confirmation fails, abort and throw an IllegalArgument exception
       */
      try
      {
         this.confirmWeightsInputFile(weightsInputReader);     // Confirm the file's network sizes match
      } // try
      catch (Exception exception)                              // Rethrow a sub-method exception
      {
         weightsInputReader.close();                           // Close the scanner
         throw (exception); 
      } // catch (Exception exception)                                                                  
      
      /*
       * Proceed to read weights post-confirmation
       */
      try
      {
         this.loadFileWeights(weightsInputReader);             // Read the weights from the file
      } // try
      catch (Exception exception)                              // Rethrow a sub-method exception
      {
         throw (exception);
      } // catch (Exception exception) 
      finally
      {
         weightsInputReader.close();                           // Always close the scanner
      } // finally
         
      return;
   } // private void readWeightsFromFile(File weightsInputFile)
   
   /*
    * Confirms a weights input file by reading the provided network sizes
    * Does not close the scanner
    * 
    * parameters: weightsInputSizesReader, the scanner to read the weights input file's provided sizes
    * preconditions: weightsInputSizesReader is initialized to a scanner
    * postconditions: If the scanner's next (2*NUM_LAYERS + 3) tokens consist of a valid weights file's provided network sizes, 
    *                 do nothing other than having advanced the scanner's position (2*NUM_LAYERS + 3) tokens forward.
    *                 Otherwise, throw an IllegalArgumentException (if the file is too short, has a mismatched data type, or provides mismatching sizes)
    *                 The scanner is not closed in this function.
    */
   private void confirmWeightsInputFile(Scanner weightsInputSizesReader) throws IllegalArgumentException
   {
      weightsInputSizesReader.useDelimiter(":|\\n|,|-");                         // Use the colon newline, comma, and hyphen as delimiters 
      
      /*
       * Store the provided number of layers and layer sizes
       */
      int fileNumLayers;
      int[] fileLayerSizes;
      
      /*
       * Read in the network sizes provided by the weights file
       */
      try
      {
         /*
          * Read the provided number of layers
          */
         weightsInputSizesReader.next();                                         // Reads label "NUM_LAYERS"
         fileNumLayers = weightsInputSizesReader.nextInt();                      // Reads the file's number of layers
                  
         /*
          * Read the provided size layers
          */
         weightsInputSizesReader.next();                                         // Reads label "LAYER_SIZES"

         fileLayerSizes = new int[fileNumLayers];                                // Allocate the array storing the file's network sizes

         for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)                       // Loops for each layer size
         {
            fileLayerSizes[alpha] = weightsInputSizesReader.nextInt();           // Reads the network sizes in order
         } // for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
      } // try
      
      catch (InputMismatchException inputMismatchException)                      // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid weights input file. The weights input file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)                      // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid weights input file. The weights input file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      /*
       * Confirm that the provided number of layers matches the actual network's number of layers. 
       * If they don't, throw an exception.
       */
            
      if (fileNumLayers != this.NUM_LAYERS)
      {
         String longExceptionMessage = "Invalid weights input file. The provided number of layers ";
         longExceptionMessage += "(" + fileNumLayers + ")";
         longExceptionMessage += " does not match the network number of layers ";
         longExceptionMessage += "(" + this.NUM_LAYERS + ").";
         
         throw (new IllegalArgumentException(longExceptionMessage));
      } // if (fileNumLayers != this.NUM_LAYERS)
      
      /*
       * Check that each provided layer sizes match the actual layer sizes
       */
      boolean layerSizesMatch = true;                                            // Is set to false if the layer sizes do not match             
      alpha = 0;
      
      while (layerSizesMatch && alpha < this.NUM_LAYERS)
      {
         layerSizesMatch = (fileLayerSizes[alpha] == this.LAYER_SIZES[alpha]);
         ++alpha;
      } // while (layerSizesMatch && alpha < this.NUM_LAYERS)
      
      /*
       * If the layer sizes do not match, throw an exception
       */
      if (!layerSizesMatch)
      {
         /*
          * Build the exception string
          */
         String longExceptionMessage = "Invalid weights input file. The provided layer sizes (";
         
         String fileLayerSizesString = "";         
         String actualLayerSizesString = "";
         
         /*
          * Add the terms to the string preceded with a dash (including the first element)
          */
         for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
         {
            fileLayerSizesString += ("-" + fileLayerSizes[alpha]);
            actualLayerSizesString += ("-" + this.LAYER_SIZES[alpha]);
         } // for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha) 
         
         /*
          * Remove the first dash (the first character in the string) and add to the overall exception message
          */
         longExceptionMessage += fileLayerSizesString.substring(1) + ") does not match the network layer sizes (";
         longExceptionMessage += actualLayerSizesString.substring(1) + ").";
         
         /*
          * Throw the exception
          */
         throw (new IllegalArgumentException(longExceptionMessage));
      } // if (!layerSizesMatch)
      
      return;
   } // private void confirmWeightsInputFile(Scanner weightsInputSizesReader) throws IllegalArgumentException

   /*
    * Loads in file weights by reading the provided weights
    * Does not close the scanner
    * 
    * parameters: weightsReader, the scanner to read the weights input file's provided weights
    * preconditions: weightsReader is initialized to a scanner
    * postconditions: If the scanner's next (total number of weights + NUM_LAYERS) tokens consist of a valid weights file's provided weight values, 
    *                 read the provided weights into the network's weight arrays.
    *                 The scanner's position therefore advances (total number of weights + NUM_LAYERS) tokens forward.
    *                 Otherwise, throw an IllegalArgumentException (if the file is too short or has a mismatched data type).
    *                 The scanner is not closed in this function. 
    */
   private void loadFileWeights(Scanner weightsReader) throws IllegalArgumentException
   {
      weightsReader.useDelimiter(":|\\n|,");                                  // Use the colon, newline, and comma as delimiters 
            
      /*
       * Read the weights into the weight arrays.
       * Use generalized indices since all the weights can be read identically
       */
      try
      {
         for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)                // Loop over the layers for the synapse source
         {
            weightsReader.next();                                             // Reads an empty token separating layers
                        
            for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)            // Loop through the current layer (synapse source)
            {
               for (gamma = 0; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)  // Loop through the next layer (synapse destination)
               {
                  this.w[alpha][beta][gamma] = weightsReader.nextDouble();    // Reads the corresponding weight value
               } // for (gamma = 0; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)
            } // for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta) 
         } // for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)
      } // try
      
      catch (InputMismatchException inputMismatchException)                   // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid weights input file. The weights input file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)                   // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid weights input file. The weights input file contains too few arguments."));
      } //  catch (NoSuchElementException noSuchElementException)
      
      return;
   } // private void loadFileWeights(Scanner weightsReader) throws IllegalArgumentException
   
   /*
    * Allocates unit arrays
    * 
    * preconditions: NUM_LAYERS is set to an appropriate positive integer
    *                and LAYER_SIZES is set to an appropriate positive integer array of size NUM_LAYERS
    * postconditions: the unit arrays are allocated
    */
   private void allocateUnits() 
   {
      /*
       * Allocate the overall array to hold all the layers
       */
      this.a = new double[this.NUM_LAYERS][];
      
      /*
       * Allocate each layer
       */
      for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
      {
         this.a[alpha] = new double[this.LAYER_SIZES[alpha]];
      } // for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
      
      return;
   } // private void allocateUnits()
   
   /*
    * Allocate the target array
    * 
    * preconditions: the number of output units (this.LAYER_SIZES[this.NUM_LAYERS - 1])
    *                is set to an appropriate positive integer.
    * postconditions: the desired target array is allocated.
    */
   private void allocateTargets()
   {
      this.T = new double[this.LAYER_SIZES[this.NUM_LAYERS - 1]];
      
      return;
   } // private void allocateTargets()
   
   /*
    * Allocate the weights array
    * 
    * preconditions: NUM_LAYERS is set to an appropriate positive integer 
    *                and LAYER_SIZES is set to an appropriate positive integer array of size NUM_LAYERS
    * postconditions: the weights array is allocated
   */
   private void allocateWeights()
   {    
      w = new double[this.NUM_LAYERS - 1][][];                    // Allocate the 3D array
      
      /*
       * Allocate each layer
       */
      for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)
      {
         this.w[alpha] = new double[this.LAYER_SIZES[alpha]][this.LAYER_SIZES[alpha + 1]];
      } // for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
      
      return;
   } // private void allocateWeights()
      
   /*
    * Loads weights from an input array
    * 
    * parameters: new_w, the input weight array
    * preconditions: the weight array and the input array have been allocated as a 3D array consisting of 
    *                (NUM_LAYERS - 1) 2D arrays whose sizes correspond with the network with layer sizes LAYER_SIZES
    * postconditions: the internal weight array matches the input weight array
    */
   public void loadWeights(double new_w[][][])
   {
      /*
       * Use generalized indices since all the weights can be loaded identically
       */
      for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)                // Loop over the layers for the synapse source
      {         
         for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)            // Loop through the current layer (synapse source)
         {
            for (gamma = 0; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)  // Loop through the next layer (synapse destination)
            {
               this.w[alpha][beta][gamma] = new_w[alpha][gamma][beta];     // Copy the corresponding weight value
            } // for (gamma = 0; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)
         } // for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
      } // for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)
      
      return;
   } // public void loadWeights(double new_w[][][])
   
   /*
    * Returns a random double within the network's specified range
    * 
    * return: a random double in the interval [RANDOM_WEIGHT_MIN, RANDOM_WEIGHT_MAX)
    */
   private double randomDouble()
   {
      double range = this.RANDOM_WEIGHT_MAX - this.RANDOM_WEIGHT_MIN;
      
      return (this.RANDOM_WEIGHT_MIN + range * Math.random());
   } // private static double randomDouble()
   
   /*
    * Randomizes weights to doubles within the network's specific range
    * 
    * preconditions: the weight array and the input array have been allocated as a 3D array consisting of 
    *                (NUM_LAYERS - 1) 2D arrays whose sizes correspond with the network with layer sizes LAYER_SIZES 
    * postconditions: the weight array is populated with random doubles in the interval [RANDOM_WEIGHT_MIN, RANDOM_WEIGHT_MAX)
    */
   public void randomizeWeights()
   {
      /*
       * Use generalized indices since all the weights can be loaded identically
       */
      for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)                // Loop over the layers for the synapse source
      {         
         for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)            // Loop through the current layer (synapse source)
         {
            for (gamma = 0; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)  // Loop through the next layer (synapse destination)
            {
               this.w[alpha][beta][gamma] = this.randomDouble();           // Copy the corresponding weight value
            } // for (gamma = 0; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)
         } // for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
      } // for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)
      
      return;  
   } // public void randomizeWeights()
   
   /*
    * Allocate the necessary Theta arrays
    * 
    * preconditions: NUM_LAYERS is set to an appropriate positive integer,
    *                and LAYER_SIZES is an appropriate positive integer array of size NUM_LAYERS
    * postconditions: the hidden layer Theta arrays are allocated as a 2D array of size NUM_LAYERS.
    *                 The input and output layer theta arrays are not allocated: their rows are included as empty to simplify indexing.
    */
   private void allocateThetas()
   {
      this.Theta = new double[this.NUM_LAYERS][];                 // Overall array
      
      /*
       * Input Thetas never need to be stored: put a placeholder for simplified indexing
       */
      alpha = 0;
      this.Theta[alpha] = null;
      
      /*
       * Allocate hidden Thetas
       */
      for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++alpha)       // Loops over hidden layers
      {
         this.Theta[alpha] = new double[this.LAYER_SIZES[alpha]];         
      } // for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++alpha)
      
      /*
       * Output Thetas never need to be stored: put a placeholder for simplified indexing
       */
      alpha = this.NUM_LAYERS - 1;
      this.Theta[alpha] = null;
      
      return;
   } // private void allocateThetas()

   /*
    * Allocate the necessary Psi arrays
    * 
    * preconditions: NUM_LAYERS is set to an appropriate positive integer,
    *                and LAYER_SIZES is an appropriate positive integer array of size NUM_LAYERS
    * postconditions: non-first hidden and output Psi arrays are allocated as a 2D array of size NUM_LAYERS.
    *                 The input and first hidden layer Psi arrays are not allocated: their rows are included as empty to simplify indexing.
    */
   private void allocatePsis()
   {
      this.Psi = new double[this.NUM_LAYERS][];                // Overall array
      
      /*
       * Input layer Psis never need to be stored: put a placeholder for simplified indexing
       */
      alpha = 0;
      this.Psi[alpha] = null;
      
      /*
       * First hidden layer Psis also never need to be stored: put a placeholder for simplified indexing
       */
      alpha = 1;
      this.Psi[alpha] = null;
      
      /*
       * Allocate non-first hidden and output Thetas
       */
      for (alpha = 2; alpha < this.NUM_LAYERS; ++alpha)        // Loops over middle layers right of the first and the output
      {
         this.Psi[alpha] = new double[this.LAYER_SIZES[alpha]];
      } // for (alpha = 2; alpha < this.NUM_LAYERS; ++alpha)
      
      return;
   } // private void allocatePsis()
   
   /*
    * Loads new inputs into the input units.
    * 
    * parameters: new_a are the new inputs
    * preconditions: the argument new_a is of length LAYER_SIZES[0]
    * postconditions: the argument new_a values are copied into the internal input units
    * 
    * BOGO: not copied anymore
    */
   private void loadInputs(double[] new_a)
   {
      alpha = 0;                       // Select the input layer
            
//      for (m = 0; m < this.LAYER_SIZES[alpha]; ++m)
//      {
//         this.a[alpha][m] = new_a[m];
//      } // for (m = 0; m < this.LAYER_SIZES[alpha]; ++m)
//      
      this.a[alpha] = new_a;
      
      return;
   } // private void loadInputs(double[] new_a)
      
   /*
    * Loads new targets into stored targets
    * 
    * parameters: new_T is the new targets
    * preconditions: new_T is of length LAYER_SIZES[NUM_LAYERS - 1]
    * postconditions: the argument new_T is copied into the internal targets
    * 
    * BOGO: not copied anymore
    */
   private void loadTargets(double[] new_T)
   {
//      alpha = this.NUM_LAYERS - 1;     // Select the output layer
      
//      for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
//      {
//         this.T[i] = new_T[i];
//      } // for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
      
      this.T = new_T;
      
      return;
   } // private void loadTargets(double[] new_T)
   
   /*
    * Returns the output unit of the network
    * 
    * return: the values of the output units
    */
   public double[] getOutputs()
   {
      alpha = this.NUM_LAYERS - 1;
      
      return this.a[alpha];
   } // public double[] getOutputs()
   
   /*
    * Extracts a set of input sets from a super input file that contains the file names of mini input files
    */
   public double[][] extractInputSetFromSuperFile(File superFile) throws FileNotFoundException, IllegalArgumentException
   {
      double[][] inputSet = null;
      
      Scanner superReader;
      int numMembers;
      
      /*
       * Open the scanner and verify the file path object identifies an existing file
       */
      try
      {
         superReader = new Scanner(superFile);
      } // try
      
      catch (FileNotFoundException fileNotFoundException)         // If the file is not found, label and throw an exception
      {
         throw (new FileNotFoundException("Super input file not found: " + fileNotFoundException.getMessage()));
      } // catch (FileNotFoundException fileNotFoundException)
      
      /*
       * Try to read the number of input units provided by the file and test for confirmation
       */
      try
      {
         superReader.useDelimiter(":|\\n");                       // Use colon or newline
         
         superReader.next();                                      // Read the label "NUM_MEMBERS"
         numMembers = superReader.nextInt();                      // Read the file's provided number of input members
        
         /*
          * Allocate the input array with numMembers rows and actualNumInputUnits columns
          */
         inputSet = new double[numMembers][this.getNumInputUnits()];
         
         /*
          * Read the file into the inputs
          */
         superReader.useDelimiter(":|\\n|,");                     // Use colon, newline, or comma
         
         superReader.next();                                      // Read an empty token separating input confirmation/file length and the input members
                  
         for (int member = 0; member < numMembers; ++member)      // Loop over members (lines in the file)
         {
            inputSet[member] = this.extractInputMemberFromMiniFile(new File(superReader.next())); // Reads the corresponding input member file
         } // for (int member = 0; member < numMembers; ++member)
      } // try
      
      catch (InputMismatchException inputMismatchException)       // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid super input set file. The super file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)       // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid super input set file. The super file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      finally
      {
         superReader.close();                                    // Always close the scanner
      } // finally
      
      
      return inputSet;
   } // public double[][] extractInputSetFromSuperFile(File superFile)
   
   /*
    * Extracts an input member from a mini input file
    */
   public double[] extractInputMemberFromMiniFile(File miniFile) throws FileNotFoundException, IllegalArgumentException
   {
      double[] inputMember = null;
    
      Scanner miniReader;
      int fileNumInputUnits;
      
      /*
       * The actual number of input units
       */
      int actualNumInputUnits = this.getNumInputUnits();
      
      /*
       * Open the scanner and verify the file path object identifies an existing file
       */
      try
      {
         miniReader = new Scanner(miniFile);
      } // try
      
      catch (FileNotFoundException fileNotFoundException)         // If the file is not found, label and throw an exception
      {
         throw (new FileNotFoundException("Input set file not found: " + fileNotFoundException.getMessage()));
      } // catch (FileNotFoundException fileNotFoundException)
      
      /*
       * Try to read the number of input units provided by the file and test for confirmation
       */
      try
      {
         miniReader.useDelimiter(":|\\n");                        // Use colon, newline, or comma
         
         miniReader.next();                                       // Read the label "NUM_INPUT_UNITS"
         fileNumInputUnits = miniReader.nextInt();                // Read the file's provided number of input units
         
         if (fileNumInputUnits != actualNumInputUnits)            // Check if the provided number of output units matches the actual number
         {
            String longExceptionMessage = "Invalid input set file. The provided number of input units";
            longExceptionMessage += " (" + fileNumInputUnits + ") does not match the network's number of input units (" + actualNumInputUnits + ").";
            throw (new IllegalArgumentException(longExceptionMessage));
         } // if (fileNumInputUnits != actualNumInputUnits)
         
         /*
          * Allocate the input member with actualNumInputUnits columns
          */
         inputMember = new double[actualNumInputUnits];
         
         /*
          * Read the file into the inputs
          */
         miniReader.useDelimiter(":|\\n|,");                      // Use colon, newline, or comma
                  
         for (i = 0; i < actualNumInputUnits; ++i)                // Loop over the inputs for each member
         {
               inputMember[i] = miniReader.nextDouble();          // Reads the corresponding input value
         } // for (m = 0; m < actualNumInputUnits; ++m)
      } // try
      
      catch (InputMismatchException inputMismatchException)       // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid input set file. The inputs file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)       // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid input set file. The inputs file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      finally
      {
         miniReader.close();                                    // Always close the scanner
      } // finally
      
      
      return inputMember;
   } // public double[] extractInputMemberFromMiniFile(File miniFile) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Extracts a set of inputs from a file
    * Confirms that the file's number of input units matches the number of input units 
    * 
    * parameters: inputSetFile, the file path object to extract the input set from
    * preconditions: inputSetFile is initialized to a file path
    * postconditions: If the given file path object does not identify an existing file, 
    *                 aborts and throws a FileNotFoundException.
    *                 If the given file path object identifies an improperly formatted input set file, 
    *                 aborts and throws an IllegalArgumentException.
    *                 If the given file path object identifies an input set file with a non-matching number of input units, 
    *                 aborts and throws an IllegalArgumentException.
    * return: the set of inputs in array form, with each row representing one input member
    */
   public double[][] extractInputSetFromFile(File inputSetFile) throws FileNotFoundException, IllegalArgumentException
   {
      double[][] inputSet = null;                       // Stores the input set extracted from the file
      
      Scanner inputsReader;
      int fileNumInputUnits;
      int numMembers;

      /*
       * The actual number of input units
       */
      int actualNumInputUnits = this.getNumInputUnits();
      
      /*
       * Open the scanner and verify the file path object identifies an existing file
       */
      try
      {
         inputsReader = new Scanner(inputSetFile);
      } // try
      
      catch (FileNotFoundException fileNotFoundException)         // If the file is not found, label and throw an exception
      {
         throw (new FileNotFoundException("Input set file not found: " + fileNotFoundException.getMessage()));
      } // catch (FileNotFoundException fileNotFoundException)
      
      /*
       * Try to read the number of input units provided by the file and test for confirmation
       */
      try
      {
         inputsReader.useDelimiter(":|\\n");                      // Use colon or newline
         
         inputsReader.next();                                     // Read the label "NUM_INPUT_UNITS"
         fileNumInputUnits = inputsReader.nextInt();              // Read the file's provided number of input units
         
         if (fileNumInputUnits != actualNumInputUnits)            // Check if the provided number of output units matches the actual number
         {
            String longExceptionMessage = "Invalid input set file. The provided number of input units";
            longExceptionMessage += " (" + fileNumInputUnits + ") does not match the network's number of input units (" + actualNumInputUnits + ").";
            throw (new IllegalArgumentException(longExceptionMessage));
         } // if (fileNumInputUnits != actualNumInputUnits)
         
         /*
          * Otherwise, read the number of input members
          */
         inputsReader.next();                                     // Read the label "NUM_MEMBERS"
         numMembers = inputsReader.nextInt();                     // Read the number of input members
                 
         /*
          * Allocate the input array with numMembers rows and actualNumInputUnits columns
          */
         inputSet = new double[numMembers][actualNumInputUnits];
         
         /*
          * Read the file into the inputs
          */
         inputsReader.useDelimiter(":|\\n|,");                    // Use colon, newline, or comma
         
         inputsReader.next();                                     // Read an empty token separating input confirmation/file length and the input members
         
         for (int member = 0; member < numMembers; ++member)      // Loop over members (lines in the file)
         {  
            for (m = 0; m < actualNumInputUnits; ++m)             // Loop over the inputs for each member
            {
               inputSet[member][m] = inputsReader.nextDouble();   // Reads the corresponding input value
            } // for (m = 0; m < actualNumInputUnits; ++m)
         } // for (int member = 0; member < numMembers; ++member)
      } // try
      
      catch (InputMismatchException inputMismatchException)       // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid input set file. The inputs file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)       // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid input set file. The inputs file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      finally
      {
         inputsReader.close();                                    // Always close the scanner
      } // finally
     
      return inputSet;
   } // private double[][] extractInputSetFromFile(File inputSetFile) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Extracts a set of targets from a file
    * Confirms that the file's number of output units matches the number of output units 
    * 
    * parameters: targetSetFile, the file path object to extract the target set from
    * postconditions: If the given file path object does not identify an existing file, 
    *                 aborts and throws a FileNotFoundException.
    *                 If the given file path object identifies an improperly formatted target set file, 
    *                 aborts and throws an IllegalArgumentException.
    *                 If the given file path object identifies a target set file with a non-matching number of output units, 
    *                 aborts and throws an IllegalArgumentException.
    * return: the set of inputs in array form, with each row representing one input member
    */
   public double[][] extractTargetSetFromFile(File targetSetFile) throws FileNotFoundException, IllegalArgumentException
   {
      double[][] targetSet = null;                                   // Stores the target set extracted from the file
      
      Scanner targetSetReader;
      int fileNumOutputUnits;
      int numMembers;
      
      /*
       * The actual number of output units
       */
      int actualNumOutputUnits = this.getNumOutputUnits();
      
      /*
       * Open the scanner and verify the file path object identifies an existing file
       */
      try
      {
         targetSetReader = new Scanner(targetSetFile);
      } // try
      
      catch (FileNotFoundException fileNotFoundException)            // If the file is not found, label and throw an exception
      {
         throw (new FileNotFoundException("Target set file not found: " + fileNotFoundException.getMessage()));
      } // catch (FileNotFoundException fileNotFoundException)
      
      /*
       * Try to read the number of output units provided by the file and test for confirmation
       */
      try
      {
         targetSetReader.useDelimiter(":|\\n");                      // Use colon or newline
         
         targetSetReader.next();                                     // Read the label "NUM_OUTPUT_UNITS"
         fileNumOutputUnits = targetSetReader.nextInt();             // Read the file's provided number of output units
         
         if (fileNumOutputUnits != actualNumOutputUnits)             // Check if the provided number of output units matches the actual number
         {
            String longExceptionMessage = "Invalid target set file. The provided number of output units";
            longExceptionMessage += " (" + fileNumOutputUnits + ") does not match the network's number of target units (" + fileNumOutputUnits + ").";
            throw (new IllegalArgumentException(longExceptionMessage));
         } // if (fileNumOutputUnits != actualNumOutputUnits)  
         
         /*
          * Otherwise, read the number of target members
          */
         targetSetReader.next();                                     // Read the label "NUM_MEMBERS"
         numMembers = targetSetReader.nextInt();                     // Read the number of target members
                 
         /*
          * Allocate the target array with numMembers rows and actualNumOutputUnits columns
          */
         targetSet = new double[numMembers][actualNumOutputUnits];
         
         /*
          * Read the file into the weights array
          */
         targetSetReader.useDelimiter(":|\\n|,");                    // Use colon, newline, or comma
         
         targetSetReader.next();                                     // Read an empty token separating target confirmation/file length and the target members
         
         for (int member = 0; member < numMembers; ++member)         // Loop over members (lines in the file)
         {  
            for (i = 0; i < actualNumOutputUnits; ++i)               // Loop over the targets for each member
            {
               targetSet[member][i] = targetSetReader.nextDouble();  // Reads the corresponding target value
            } // for (i = 0; i < actualNumOutputUnits; ++u)   
         } // for (int member = 0; member < numMembers; ++member)
         
      } // try
      
      catch (InputMismatchException inputMismatchException)          // If the scanner reads a wrong data type
      {
         throw (new IllegalArgumentException("Invalid target set file. The target set file contains a mismatched data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)          // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid target set file. The target set file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)
      
      finally
      {
         targetSetReader.close();                                    // Always close the scanner
      } // finally
     
      return targetSet;
   } // private double[][] extractTargetSetFromFile(File targetSetFile) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Run the network on the given inputs WITHOUT calculating training details
    * Purely for fast and memory-efficient execution
    * 
    * parameters: inputs are the new inputs
    * preconditions: inputs is of length LAYER_SIZES[0], the unit and weight arrays are allocated, and the weights are set 
    * postconditions: the internal input units match the parameter inputs, 
    *                 and the hidden and output units are calculated appropriately. 
    *                 Thetas and Psis go unchanged.
    * return: the network outputs
    */
   public double[] runOnMember(double[] inputs)
   {
      this.loadInputs(inputs);
      this.executeWithoutDetails();
      
      return this.getOutputs();
   } // public double[] runOnMember(double[] inputs)
   
   /*
    * Run the network on the given input set WITHOUT calculating training details
    * Purely for fast and memory-efficient execution
    * 
    * parameters: inputSet is the set of new inputs
    * preconditions: inputSet has LAYER_SIZES[0] columns, the unit and weight arrays are allocated, and the weights are set 
    * postconditions: the internal input units match the last member of the parameter input set, 
    *                 and the hidden and output units are calculated appropriately. 
    *                 Thetas and Psis go unchanged.
    * return: the set of network outputs after running the network on the provided input set. Order is preserved.
    */
   public double[][] runOnSet(double[][] inputSet)
   {
      int numMembers = inputSet.length;
      double[][] outputSet = new double[numMembers][];
      
      for (int member = 0; member < numMembers; ++member)
      {
         /*
          * Run the network on each input member and hard-copy the output arrays into the master output set
          */
         outputSet[member] = this.runOnMember(inputSet[member]).clone();
      } // for (int member = 0; member < numMembers; ++member)
      
      return outputSet;
   } // public double[][] runOnSet(double[][] inputSet)
    
   /*
    * Run the network on the given input set WITHOUT calculating training details
    * Purely for fast and memory-efficient execution
    * 
    * parameters: inputSetFile is the file path object identifying the input set file.
    * preconditions: the unit and weight arrays are allocated and the weights are set 
    * postconditions: If the given file path object does not identify an existing file, aborts and throws a FileNotFoundException.
    *                 If the given file path object identifies an improperly formatted control file,
    *                 aborts and throws an IllegalArgumentException.
                      Otherwise, the internal input units match the last member of the parameter input set, 
    *                 and the hidden and output units are calculated appropriately. 
    *                 Thetas and Psis go unchanged.
    * return: the set of network outputs after running the network on the provided input set. Order is preserved.
    */
   public double[][] runOnSet(File inputSetFile) throws FileNotFoundException, IllegalArgumentException
   {
      return (this.runOnSet(this.extractInputSetFromFile(inputSetFile)));
   } // public void runOnSet(File inputSetFile) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Executes the network by evaluating hidden and output units from the input units. 
    * Training details (Thetas and Psis) are NOT collected.
    * 
    * preconditions: the input units and the network weights have been loaded appropriately
    * postconditions: updates hidden and output units based on the input units, weights, and activation function.
    */
   private void executeWithoutDetails()
   {
      /*
       * Use generalized indices (beta, gamma) since the hidden, and output unit layers can be computed identically
       */
      double localAllTheta_beta;                                                                // Temporary Theta for all layers
      
      for (alpha = 1; alpha < this.NUM_LAYERS; ++alpha)                                         // Loop over the layers for the synapse destination
      {         
         for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)                                 // Loop through the current layer (synapse destination)
         {
            localAllTheta_beta = 0.0;                                                           // Reset the local Theta
            
            for (gamma = 0; gamma < this.LAYER_SIZES[alpha - 1]; ++gamma)                       // Loop through the previous layer (synapse source)
            {
               localAllTheta_beta += this.w[alpha - 1][gamma][beta] * this.a[alpha - 1][gamma]; // Accumulate the local Theta
            } // for (gamma = 0; gamma < this.LAYER_SIZES[alpha - 1]; ++gamma)
            
            this.a[alpha][beta] = this.activationFunction(localAllTheta_beta);                  // Calculate the unit in the current layer
         } // for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
      } // for (alpha = 1; alpha < this.NUM_LAYERS; ++alpha)
      
      return;
   } // private void executeWithoutDetails()
   
   /*
    * Executes the network by evaluating hidden and output units from the input units. 
    * Theta and Psi values are also updated and used.
    * 
    * preconditions: the input units have been loaded appropriately. The Theta and output psis arrays have been allocated.
    * postconditions: updates hidden and output units, hidden Thetas, and output psis
    *                 based on the input units, weights, and activation function.
    */
   private void executeWithDetails()
   {
      /*
       * Evaluate the hidden layers, storing Thetas but NOT calculating Psis
       * Use generalized indices (beta, gamma) since the hidden layers can be calculated identically
       */
      for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++alpha)                                           // Loop over the layers for the synapse destination
      {         
         for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)                                       // Loop through the current layer (synapse destination)
         {
            this.Theta[alpha][beta] = 0.0;                                                            // Reset the stored Theta
            
            for (gamma = 0; gamma < this.LAYER_SIZES[alpha - 1]; ++gamma)                             // Loop through the previous layer (synapse source)
            {
               this.Theta[alpha][beta] += this.w[alpha - 1][gamma][beta] * this.a[alpha - 1][gamma];  // Accumulate the local Theta
            } // for (gamma = 0; gamma < this.LAYER_SIZES[alpha - 1]; ++gamma)
            
            this.a[alpha][beta] = this.activationFunction(this.Theta[alpha][beta]);                   // Calculate the unit in the current layer
         } // for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
      } // for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++alpha)

      /*
       * Evaluate the output layer, NOT storing Thetas but calculating Psis
       * Last layer: use index "i" and "j"
       */      
      double localOutputTheta_i;
      
      alpha = this.NUM_LAYERS - 1;                                                                    // Final layer
      
      for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
      {
         localOutputTheta_i = 0.0;                                                                    // Reset the local Theta
         
         for (j = 0; j < this.LAYER_SIZES[alpha - 1]; ++j)                                            // Loop through the previous layer (synapse source)
         {
            localOutputTheta_i += this.w[alpha - 1][j][i] * this.a[alpha - 1][j];                     // Accumulate the local Theta
         } // for (j = 0; j < this.LAYER_SIZES[alpha - 1]; ++j)
         
         this.a[alpha][i] = this.activationFunction(localOutputTheta_i);                              // Calculate the unit in the current layer
         
         /*
          * Calculates and stores output layer psi_i's. 
          */
         this.Psi[alpha][i] = (this.T[i] - this.a[alpha][i]);                                         // Calculates Omega_i
         this.Psi[alpha][i] *= this.activationFunctionDerivative(localOutputTheta_i);                 // Broken into two lines to prevent long line
      } // for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
      
      return;
   } // private void executeWithDetails()
   
   /*
    * Updates network weights via backpropagation
    * 
    * preconditions: the input, hidden, and output units, the hidden Thetas, and the output layer
    *                Psis are all calculated appropriately. The Psi arrays for the non-first hidden layers are allocated.
    * postconditions: the network weights are updated using the unit activations, Thetas, and Psis.
    */
   private void backpropagate()
   {
      /*
       * Loop over the second hidden layer
       * Use specific indices (j and i) since the second hidden layer is being processed
       */
      double localOmega_j;
      
      alpha = 2;                                                           // Select the second hidden layer
      
      for (j = 0; j < this.LAYER_SIZES[alpha]; ++j)                        // Loop over the second hidden layer
      {
         localOmega_j = 0.0;                                               // Reset the local Omega
         
         for (i = 0; i < this.LAYER_SIZES[alpha + 1]; ++i)                 // Loop over the output layer (right of second hidden layer)
         {
            localOmega_j += this.Psi[alpha + 1][i] * this.w[alpha][j][i];  // Accumulate the local Omega
            this.w[alpha][j][i] += this.lambda * this.a[alpha][j] * this.Psi[alpha + 1][i];
         } // for (i = 0; i < this.LAYER_SIZES[alpha + 1]; ++i)
         
         this.Psi[alpha][j] = localOmega_j * this.activationFunctionDerivative(this.Theta[alpha][j]);
      } // for (j = 0; j < this.LAYER_SIZES[alpha]; ++j)
      
      /*
       * Loop over the first hidden layer
       * Use specific indices (i, k, and m) since the first hidden layer is being processed
       */      
      double localOmega_k;
      double localPsi_k;      
      
      alpha = 1;                                                           // Select the first hidden layer
      
      for (k = 0; k < this.LAYER_SIZES[alpha]; ++k)                        // Loop over the first hidden layer
      {
         localOmega_k = 0.0;                                               // Reset the local Omega
         
         for (j = 0; j < this.LAYER_SIZES[alpha + 1]; ++j)                 // Loop over the second hidden layer
         {
            localOmega_k += this.Psi[alpha + 1][j] * this.w[alpha][k][j];  // Accumulate the local Omega
            this.w[alpha][k][j] += this.lambda * this.a[alpha][k] * this.Psi[alpha + 1][j];
         } // for (j = 0; j < this.LAYER_SIZES[alpha + 1]; ++j)
         
         localPsi_k = localOmega_k * this.activationFunctionDerivative(this.Theta[alpha][k]);
         
         /*
          * Update the input layer weights without calculating further Omegas or Psis
          */
         for (m = 0; m < this.LAYER_SIZES[alpha - 1]; ++m)
         {
            this.w[alpha - 1][m][k] += this.lambda * this.a[alpha - 1][m] * localPsi_k;
         } // for (m = 0; m < this.LAYER_SIZES[alpha - 1]; ++m)
      } // for (k = 0; k < this.LAYER_SIZES[alpha]; ++k)
      
      return;
   } // private void backpropagate()
   
   /*
    * Returns the activation function of a real number
    * 
    * parameters: x, the real number input
    * return: the hyperbolic tangent of the input
    */
   private double activationFunction(double x) 
   {
      // return 1.0/(1.0 + Math.exp(-x));  // Sigmoid   
      
      return Math.tanh(x);
   } // private double activationFunction(double x)
   
   /*
    * Returns the derivative of the activation function of a real number
    * 
    * parameters: x, the real number input
    * return: the derivative of the sigmoid of the input
    */
   private double activationFunctionDerivative(double x)
   {  
      // double f = this.activationFunction(x);
      // return f * (1.0 - f);             // Sigmoid derivative
      
      double c = Math.cosh(x);
      return 1.0/(c * c);                 // Hyperbolic tangent derivative
      
   } // private double activationFunctionDerivative(double x)
   
   /*
    * Calculates the error
    * 
    * preconditions: omega_i is calculated for all i
    * return: the total error of the network
    */
   public double getError()
   {
      double total = 0.0;
      double current_omega;
      
      alpha = this.NUM_LAYERS - 1;                     // Select output layer
      
      for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
      {
         current_omega = this.T[i] - this.a[alpha][i];         
         total += current_omega * current_omega;    
      } // for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
      
      return (total/2.0);
   } // public double getError()

   /*
    * Trains the network on a single training member
    * 
    * parameters: inputs is the set of inputs, targets is the set of intended target outputs
    * preconditions: the unit and weight arrays are allocated correctly
    * postconditions: the inputs and targets are loaded into the network,
    *                 the hidden and output units of the network are computed from the inputs, 
    *                 the weight training details are calculated,
    *                 and the weight changes are applied.
    */
   public void trainOnMember(double[] inputs, double[] targets)
   {
      this.loadInputs(inputs);
      this.loadTargets(targets);
      
      this.executeWithDetails();
      this.backpropagate();
   
      return;
   } // public void trainOnMember(double[] inputs, double[] targets)
   
   /*
    * Trains the network over a training set until enough iterations are completed or the error threshold is achieved
    * Optionally saves weights after a set number of iterations and/or at the end of training.
    * 
    * parameters: inputSet is the set of each input member, targetSet is the set of corresponding target outputs
    * preconditions: the unit and weight arrays are allocated correctly, and each inputSet element corresponds with the targetSet element of the same index
    *                If weights are to be saved, then the weight-saving file and frequency are set.
    * postconditions: the weights of the arrays are altered until either enough iterations have occurred or the largest error of the network 
    *                 over the training set (with latency) is lower than the network's threshold value. The network's units and target are populated with 
    *                 the last member in the input set and target set respectively.
    *                 If saveWeightsEvery is nonzero, the weights are written to the specified file every saveWeights iterations.
    *                 If saveWeightsAtEnd is true, writes the weights are written to specified file at the end of training.
    *                 If weights are to be saved, it is possible an IOException is thrown during file writing. Execution is then aborted.
    */
   public void trainOnSet(double[][] inputSet, double[][] targetSet) throws IOException
   {
      /*
       * Training progress variables
       */
      this.numIterations = 0;
      this.maximumSetError = 0.0;                                       // the maximum error on this iteration of the training set
      
      
      /*
       * Training set iteration
       */
      int memberIndex = 0;
      int numMembers = inputSet.length;
      
      /*
       * Whether to save weights this iteration
       * Whether to print iteration number
       */
      boolean saveOnThisIteration = false;
      boolean printOnThisIteration = false;
      
      /*
       * Start the training timer
       */
      long timeBeforeTraining = System.currentTimeMillis();
      
      /*
       * Store the error
       */
      double currentError;
      
      /*
       * Loop until either enough iterations over the entire training set are performed or 
       * the maximum error for a training member is lower than the error threshold
       */
      while (!this.maxIterationsReached && !this.errorThresholdSatisified)
      {
         /*
          * Reset the set error
          */
         maximumSetError = 0.0;
         
         /*
          * Iterate through each member in the training set once
          */
         for (memberIndex = 0; memberIndex < numMembers; ++memberIndex)
         {  
            /*
             * Train the network on the selected member.
             * inputSet[memberIndex] extracts row
             */
            this.trainOnMember(inputSet[memberIndex], targetSet[memberIndex]);
            
            /*
             * Update the maximum error
             */
            currentError = this.getError();
            if (maximumSetError < currentError)
               maximumSetError = currentError;
         } // for (memberIndex = 0; memberIndex < numMembers; ++memberIndex)
         
         /*
          * Increment the number of iterations and update the stopping flags
          */
         ++ numIterations;
         this.maxIterationsReached = (numIterations >= this.maxIterations);
         this.errorThresholdSatisified = (maximumSetError < this.errorThreshold);
         
         /*
          * Decide whether to save the weights: if saveWeightsEvery is nonzero, check if the number of completed iterations
          * is divisible by saveWeightsEvery: if it is, save the weights.
          */
         saveOnThisIteration = (this.saveWeightsEvery != 0 && numIterations % this.saveWeightsEvery == 0);
         
         if (saveOnThisIteration)
         {
            this.saveWeights();
            System.out.println("Saved weights on iteration " + numIterations + " to " + this.weightsOutputFile.getName());
         } // if (saveOnThisIteration)
         
         /*
          * Decide whether to print: if printEvery is nonzero, check if the number of completed iterations
          * is divisible by printEvery: if it is, print.
          */
         printOnThisIteration = (this.printEvery != 0 && numIterations % this.printEvery == 0);
         
         if (printOnThisIteration)
         {
            System.out.println("Completed iteration " + numIterations + " with error " + maximumSetError + " and " + (System.currentTimeMillis() - timeBeforeTraining) + " milliseconds elapsed");
         } // if (saveOnThisIteration)
         
      } // while (!this.maxIterationsReached && !this.errorThresholdSatisified)
      
      /*
       * If the weights should be saved at the end AND the last iteration didn't already save, save the weights
       */
      if (this.saveWeightsAtEnd && !saveOnThisIteration)
      {
         this.saveWeights();
         System.out.println("Saved weights at end on iteration " + numIterations + " to " + this.weightsOutputFile.getName());
      } // if (this.saveWeightsAtEnd && !saveOnThisIteration)
      
      return;
   } // public void trainOnSet(double[][] inputSet, double[][] targetSet) throws IOException
   
   /*
    * Trains the network over a training set until enough iterations are completed or the error threshold is achieved
    * Optionally saves weights after a set number of iterations and/or at the end of training.
    * 
    * parameters: inputSetFile is the file path object identifying the file containing the input members, 
    *             targetSet is the file path object identifying the file containing the corresponding target outputs
    * preconditions: the unit and weight arrays are allocated correctly, and each inputSetFile input element corresponds 
    *                with the targetSetFile target element on the same line.
    *                If weights are to be saved, then the weight-saving file and frequency are set.
    * postconditions: the weights of the arrays are altered until either enough iterations have occurred or the largest error of the network 
    *                 over the training set (with latency) is lower than the network's threshold value. The network's units and target are populated with 
    *                 the last member in the input set and target set respectively.
    *                 If the input set file or target set pile file either do not exist or are improperly formatted, throws a 
    *                 FileNotFoundException or IllegalArgumentException respectively
    *                 If saveWeightsEvery is nonzero, the weights are written to the specified file every saveWeights iterations.
    *                 If saveWeightsAtEnd is true, writes the weights are written to specified file at the end of training.
    *                 If weights are to be saved, it is possible an IOException is thrown during file writing. Execution is then aborted.
    */
   public void trainOnSet(File inputSetFile, File targetSetFile) throws FileNotFoundException, IllegalArgumentException, IOException
   {
      this.trainOnSet(this.extractInputSetFromFile(inputSetFile), this.extractTargetSetFromFile(targetSetFile));
      
      return;
   } // public void trainOnSet(File inputSetFile, File targetSetFile) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Saves the network's current weights to the network's weights output file
    * Creates the weights output file if does not exist and overwrites any existing content
    * 
    * preconditions: the network's weightsOutputFile is set
    * postconditions: if writing to the file causes an IO exception, an IO exception is thrown.
    *                 Otherwise, either create a new file at the network's provided weights output file path or overwrite
    *                 the existing weights output file with network sizes and the series of weights. This file can be used
    *                 to load weights into the network.
    */
   public void saveWeights() throws IOException
   {
      FileWriter weightsOutputWriter = null;
      
      try
      {
         this.weightsOutputFile.createNewFile();      // Creates a new file IF AND ONLY IF the file does not already exist
         
         weightsOutputWriter = new FileWriter(this.weightsOutputFile);
         
         /*
          * Write the number of layers
          */
         weightsOutputWriter.write("NUM_LAYERS:" + this.NUM_LAYERS + "\n");
         
         /*
          * Write the layer sizes starting with the input layer size and 
          * ending with the output layer size separated by hyphens
          */
         String layerSizes = "";
         
         for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
         {
            layerSizes += ("-" + this.LAYER_SIZES[alpha]); 
         } // for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
         
         /*
          * Remove the first hyphen, label, and write to the file
          */
         weightsOutputWriter.write("LAYER_SIZES:" + layerSizes.substring(1));
         
         /*
          * Write each weights layer, starting with input-hidden 1 layer and ending with hidden 2-output layer
          * Uses generalizes indices (beta and gamma) since all weight arrays can be written identically
          */
         for (alpha = 0; alpha < this.NUM_LAYERS - 1; ++alpha)
         {
            weightsOutputWriter.write("\n");             // Write a blank line separating layers
            
            for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
            {
               /*
                * Write first weight for the row (which is guaranteed to exist) preceded with a new line
                */
               gamma = 0;
               weightsOutputWriter.write("\n" + this.w[alpha][beta][gamma]);
               
               /*
                * Write the remaining weights on this row with a preceding comma
                */
               for (gamma = 1; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)
               {
                  weightsOutputWriter.write("," + this.w[alpha][beta][gamma]);
               } // for (gamma = 0; gamma < this.LAYER_SIZES[alpha + 1]; ++gamma)
               
            } // for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
         } // for (alpha = 0; alpha < this.NUM_LAYERS; ++alpha)
         
      } // try
      
      catch (IOException ioException)                 // If the file writer encounters an error
      {
         throw (new IOException("IO exception encountered while writing to output file: " + ioException.getMessage()));
      } // catch (IOException ioException)
      
      finally
      {
         weightsOutputWriter.close();                 // Always close the file writer
      } // finally
      
      return;
   } // public void saveWeights() throws IOException
   
   /*
    * Prints the input units
    * 
    * preconditions: the input units are allocated
    * postconditions: the values of the input units are printed to the console
    */
   public void printInputUnits()
   {
      System.out.println("Input units");
      
      /*
       * Print the input units
       * Use specific indices (k) to specifically print the input layer
       */
      alpha = 0;                          // Select the input layer
      
      for (k = 0; k < this.LAYER_SIZES[alpha]; ++k)
      {
         System.out.println("\ta[" + alpha + "][" + k + "] = " + this.a[alpha][k]);
      } // for (k = 0; k < this.LAYER_SIZES[alpha]; ++k)
     
      return;
   } // public void printInputUnits()
   
   /*
    * Prints the hidden units
    * 
    * preconditions: the hidden units are allocated
    * postconditions: the values of the hidden units are printed to the console
    */
   public void printHiddenUnits()
   {
      /*
       *  Loop over the hidden layers
       *  Use generalized indices (beta) to print the layers exclusively between the input and output layer
       */
      for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++alpha)
      {
         System.out.println("Hidden " + alpha + " units");
         
         for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
         {
            System.out.println("\ta[" + alpha + "][" + beta + "] = " + this.a[alpha][beta]);
         } // for (beta = 0; beta < this.LAYER_SIZES[alpha]; ++beta)
         
      } // for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++i)
      
      return;
   } // public void printHiddenUnits()
   
   /*
    * Prints the output units
    * 
    * postconditions: the value of the output units are printed to the console
    */
   public void printOutputUnits()
   {
      System.out.println("Output units");
      
      /*
       * Print the output units
       * Use specific indices (i) to print specifically the output layer
       */
      alpha = this.NUM_LAYERS - 1;                    // Select the output layer
      
      for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
      {
         System.out.println("\ta[" + alpha + "][" + i + "] = " + this.a[alpha][i]);
      } // for (i = 0; i < this.LAYER_SIZES[alpha]; ++i)
     
      return;
   } // public void printOutputUnits()
   
   /*
    * Prints every network unit
    * 
    * preconditions: the input, hidden, and output units are allocated
    * postconditions: the values of all the units are printed to the console
    */
   public void printUnits()
   {
      System.out.println("UNITS");
      this.printInputUnits();
      System.out.println();
      
      this.printHiddenUnits();
      System.out.println();
      
      this.printOutputUnits();
      
      return;
   } // public void printUnits()
   
   /*
    * Prints the training flags
    * 
    * postconditions: whether the maxIterations are reached and whether the error threshold was satisfied
    *                 for the last training session are printed to the console
   */
   public void printTrainingFlags()
   {
      System.out.println("maxIterationsReached = " + this.maxIterationsReached);
      System.out.println("errorThresholdSatisified = " + this.errorThresholdSatisified);
      
      return;
   } // public void printTrainingFlags()
   
   /*
    * Prints the training performance variables
    * 
    * postconditions: the number of iterations ran and the maximum set error for the last training session are printed to the console
    */
   public void printTrainingPerformance()
   {
      System.out.println("numIterations = " + this.numIterations);
      System.out.println("maximumSetError = " + this.maximumSetError);
      
      return;
   } // public void printTrainingPerformance()

   /*
    * Prints the training parameters
    * 
    * postconditions: the learning rate (lambda), the maximum error threshold, and the maximum number of iterations are printed to the console
    */
   public void printTrainingParameters()
   {
      System.out.println("lambda = " + this.lambda);
      System.out.println("errorThreshold = " + this.errorThreshold);
      System.out.println("maxIterations = " + this.maxIterations);
      
      return;
   } // public void printTrainingParameters()
   
   /*
    * Prints the method and the details of how the network's weights were initialized
    * 
    * postconditions: if the network was initialized by randomizing weights, the range of the randomized weights is printed to the console.
    *                 If the network was initialized by loading from a file, the name of the original weight file is printed to the console.
    */
   public void printWeightInitialization()
   {
      if (this.randomizeWeights)
      {
         System.out.println("Randomized weights in the range [" + this.RANDOM_WEIGHT_MIN + "," + this.RANDOM_WEIGHT_MAX + ")");
      } // if (this.randomizeWeights)
      
      else
      {
         System.out.println("Loaded weights from file " + this.weightsInputFile.getName());
      } // if (this.randomizeWeights)... else
      
      return;
   } // public void printRandomWeightRange()
   
   /*
    * Prints the number of units in each layer
    * 
    * postconditions: the numbers of input, hidden, and output units are each printed to the console
    */
   public void printNumUnits()
   {
      System.out.println("NUM_INPUT_UNITS = " + this.getNumInputUnits());
      
      String hiddenLayerSizes = "";
      
      /*
       * Loop over the hidden layers (second to second-last layer)
       */
      for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++alpha)
      {
         hiddenLayerSizes += ("," + this.LAYER_SIZES[alpha]);
      } // for (alpha = 1; alpha < this.NUM_LAYERS - 1; ++alpha)
      
      System.out.println("NUM_HIDDEN_UNITS = " + hiddenLayerSizes.substring(1)); // Remove the first character (unwanted comma)
      System.out.println("NUM_OUTPUT_UNITS = " + this.getNumOutputUnits());
      
      return;
   } // public void printNumUnits()
   
   /*
    * Prints the weights output configuration
    * 
    * postconditions: the weights output file name, the frequency of intermediary weight-saving, 
    *                 and whether to save weights at the end of training are printed to the console
    */
   public void printWeightsOutputConfiguration()
   {
      System.out.println("weightsOutputFile: " + this.weightsOutputFile);
      System.out.println("saveWeightsEvery: " + this.saveWeightsEvery);
      System.out.println("saveWeightsAtEnd: " + this.saveWeightsAtEnd);
      
      return;
   } // public void printWeightsOutputConfiguration()
   
   /*
    * Prints whether the network has allocated arrays for training
    * 
    * postconditions: whether the network allocated Theta_j, psi_i, and target arrays is printed to the console
    */
   public void printAllocateForTraining()
   {
      System.out.println("allocateForTraining = " + this.allocateForTraining);
      
      return;
   } // public void printAllocateForTraining()
   
   /*
    * Returns the number of input units
    * 
    * return: the number of input units for the network
    */
   public int getNumInputUnits()
   {
      return this.LAYER_SIZES[0];
   } // public void getNumInputUnits()

   
   /*
    * Returns the number of output units
    * 
    * return: the number of output units for the network
    */
   public int getNumOutputUnits()
   {
      return this.LAYER_SIZES[this.NUM_LAYERS - 1];
   } // public void getNumOutputUnits()
   
} // public class ABCDNetworkBP