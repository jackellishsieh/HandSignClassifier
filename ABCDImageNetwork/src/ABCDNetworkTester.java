/*
 * Tester for an A-B-C-D network that uses backpropagation and file IO
 * 
 * Author: Jack Hsieh
 * Date of creation: November 11, 2021
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.InputMismatchException;
import java.util.NoSuchElementException;
import java.util.Scanner;

/*
 * Suite for testing an A-B-C-D network using the control file provided by the first command line argument.
 * If no command line argument is provided the default control file "./controlFiles/defaultControlFile.txt" is used.
 * Constructs an A-B-C-D network of specified size and weight values using a network configuration file
 * The testing suite either runs or trains the network on a specified set of input FILES from a SUPER FILE
 */
public class ABCDNetworkTester
{  
   /*
    * Control file name
    */
   private static String defaultControlFilename = "controlFiles/defaultControlFile.txt";
   
   private static String currentControlFilename;
   
   /*
    * Column formatting string
    */
   private static String smallColumnFormat = "%-10s";
   private static String mediumColumnFormat = "%-15s";
   private static String bigColumnFormat = "%-30s";
   
   /*
    * The network constructed using the network configuration file provided by the control file.
    * Either trained or ran
    */
   private static ABCDNetwork testNetwork;
   
   /*
    * Boolean flag for deciding between a training test and running test. Read from the control file.
    */
   private static boolean doTrainNotRun;
   
   /*
    * The network configuration, input set, and target set files read from the control file
    * The input and target set files are separate from the network
    */
   private static File networkConfigurationFile;
   private static File inputSetFile;
   private static File targetSetFile;
   
   /*
    * Demonstrates that Java uses row-major order, not column-major order (technically via Iliffe vectors).
    * 
    * postconditions: information is printed to the console that demonstrates Java row extraction extracts rows for 2D and 3D arrays
    */
   public static void testArrayOrder()
   {
      System.out.println("\n2D array");

      /*
       * Performs 1D-array row extraction from a 2D array
       */
      int[][] array2D = new int[][] {{1,2,3,4},{5,6,7,8}};
      
      int[] array1D;
      
      /*
       * Prints every element in the 2D array
       */
      for (int i = 0; i < 2; ++i)
      {
         for (int j = 0; j < 4; ++j)
         {
            System.out.println("array2D[" + i + "][" + j + "] = " + array2D[i][j]);
         }
      }
      System.out.println();
      
      /*
       * Extracts 1D rows 
       */
      array1D = array2D[0];
      
      for (int k = 0; k < array1D.length; ++k)
      {
         System.out.println("(array2D[0])[" + k + "] = " + array1D[k]);
      }
      System.out.println();
      
      array1D = array2D[1];
      
      for (int k = 0; k < array1D.length; ++k)
      {
         System.out.println("(array2D[1])[" + k + "] = " + array1D[k]);
      }
      
      
      /*
       * Performs 2D-array row extraction from a 3D array
       */
      System.out.println("\n3D array");
      
      int[][][] array3D = new int[][][] { { {1,2}, {3,4}}, {{5,6},{7,8}} };
      
      /*
       * Prints every element in the 3D array
       */
      for (int i = 0; i < 2; ++i)
      {
         for (int j = 0; j < 2; ++j)
         {
            for (int k = 0; k < 2; ++k)
            {
               System.out.println("array3D[" + i + "][" + j + "][" + k + "] = " + array3D[i][j][k]);
            }
         }
      } // for (int i = 0; i < 2; ++i)
      System.out.println();
      
      /*
       * Extracts 2D arrays
       */
      array2D = array3D[0];
      
      for (int j = 0; j < 2; ++j)
      {
         for (int k = 0; k < 2; ++k)
         {
            System.out.println("(array3D[0])[" + j + "][" + k + "] = " + array2D[j][k]);
         }
      }
      System.out.println();
      
      
      array2D = array3D[1];
      for (int j = 0; j < 2; ++j)
      {
         for (int k = 0; k < 2; ++k)
         {
            System.out.println("(array3D[1])[" + j + "][" + k + "] = " + array2D[j][k]);
         }
      }
      System.out.println();
      
      return;
   } // public static void testArrayOrder()

   /*
    * Prints an output table of the network using the provided inputs
    * Identical to printComparisonTable but does NOT include target outputs
    * Uses super set input: rather than listing each input, lists the number of the member
    * 
    * parameters: the network to print an output table of and the input set to run the network on
    * preconditions: the provided network and input set are allocated.
    *                The number of columns of the input set matches the number of network input units.
    * postconditions: prints an output table presenting the network outputs using the given inputs.
    */
   private static void printOutputTable(ABCDNetwork network, double[][] inputSet)
   {
      /*
       * Get outputs from the network using the current input set
       */
      double[][] outputSet = network.runOnSet(inputSet);
      
      /*
       * Get the network sizes
       */
      int NUM_OUTPUT_UNITS = network.getNumOutputUnits();
      
      /*
       * Print input headers 
       */
      System.out.printf(smallColumnFormat, "Member #");
      
      /*
       * Print network output headers
       */
      for (int i = 0; i < NUM_OUTPUT_UNITS; ++i)
      {
         System.out.printf(smallColumnFormat, "F_i[" + i + "]");
      }
      System.out.println();
      
      /*
       * Print each case.
       */
      for (int member = 0; member < inputSet.length; ++member)
      {
         /*
          * Print number of the element
          */
         System.out.printf(smallColumnFormat, "Member " + member);
         
         /*
          * Print network outputs
          */
         for (int i = 0; i < NUM_OUTPUT_UNITS; ++i)
         {
            System.out.printf(smallColumnFormat, outputSet[member][i]);
         }
         
         System.out.println();
      } // for (int member = 0; member < inputSet.length; ++member)
      
      return;
   } // private static void printOutputTable(ABCDNetworkBP network, double[][] inputSet)
   

   /*
    * Prints an comparison table of the network using the provided inputs
    * Includes target outputs
    * Uses super set input: rather than listing each input, lists the number of the member
    * 
    * TODO
    * parameters: the network to print an output table of and the input set to run the network on
    * preconditions: the provided network and input set are allocated.
    *                The number of columns of the input set matches the number of network input units.
    * postconditions: prints an output table presenting the network outputs using the given inputs.
    */
   private static void printComparisonTable(ABCDNetwork network, double[][] inputSet, double[][] targetSet)
   {
      /*
       * Get outputs from the network using the current input set
       */
      double[][] outputSet = network.runOnSet(inputSet);
      
      /*
       * Get the network sizes
       */
      int NUM_OUTPUT_UNITS = network.getNumOutputUnits();
      
      /*
       * Print input headers 
       */
      System.out.printf(mediumColumnFormat, "Member #");
      
      /*
       * Print alternating target and output headers
       */
      for (int i = 0; i < NUM_OUTPUT_UNITS; ++i)
      {
         System.out.printf(smallColumnFormat, "T_i[" + i + "]");
         System.out.printf(bigColumnFormat, "F_i[" + i + "]");
      }
      System.out.println();
      
      /*
       * Print each case.
       */
      for (int member = 0; member < inputSet.length; ++member)
      {
         /*
          * Print number of the element
          */
         System.out.printf(mediumColumnFormat, "Member " + member);
         
         /*
          * Print alternating target and output values
          */
         for (int i = 0; i < NUM_OUTPUT_UNITS; ++i)
         {
            System.out.printf(smallColumnFormat, targetSet[member][i]);
            System.out.printf(bigColumnFormat, outputSet[member][i]);
         }
         
         System.out.println();
      } // for (int member = 0; member < inputSet.length; ++member)
      
      return;
   } // private static void printOutputTable(ABCDNetworkBP network, double[][] inputSet)
   
   
   
   /*
    * Opens and reads the control file.
    * Loads the testing mode, network configuration file, input set file, and target set file into the tester.
    * 
    * parameters: controlFile, the file path identifying the control file
    * postconditions: If the given file path object identifies a valid control file, save whether to train or run the network and 
    *                 network configuration, input set, and target set files.
    *                 If the given file path object does not identify an existing file, throw a FileNotFoundException.
    *                 If the given file path object identifies an improperly formatted control file. throw an IllegalArgumentException.
    */
   private static void readControlFile(File controlFile) throws FileNotFoundException, IllegalArgumentException
   {      
      Scanner controlFileReader;
      
      /*
       * Open the scanner and verify the file path object identifies an existing file
       */
      try
      {
         controlFileReader = new Scanner(controlFile);
         
      } // try
      catch (FileNotFoundException fileNotFoundException)                                       // If the file is not found, label and throw an exception
      {
         throw (new FileNotFoundException("Control file not found: " + fileNotFoundException.getMessage())); 
      } // catch (FileNotFoundException fileNotFoundException) 
      
      /*
       * Read the control file. 
       * Throw an IllegalArgument exception with an appropriate description if the control file's content is invalid 
       * (i.e. too short or wrong data type). 
       */
      try
      {
         controlFileReader.useDelimiter(":|\\n");
         
         /*
          * Scan each element of the control file
          */
         controlFileReader.next();                                                              // Read the label "doTrainNotRun"
         ABCDNetworkTester.doTrainNotRun = controlFileReader.nextBoolean();                     // Read the doNotTrainNotRun boolean
         
         controlFileReader.next();                                                              // Read the label "networkConfigurationFilename"
         String networkConfigurationFilename = controlFileReader.next();                        // Read the network configuration filename string
         ABCDNetworkTester.networkConfigurationFile = new File(networkConfigurationFilename);   // Store the network configuration file
         
         controlFileReader.next();                                                              // Read the label "inputSetFile"
         String inputSetFilename = controlFileReader.next();                                    // Read the input set filename string
         ABCDNetworkTester.inputSetFile = new File(inputSetFilename);                           // Store the input set file
         
         controlFileReader.next();                                                              // Read the label "targetSetFile"
         String targetSetFilename = controlFileReader.next();                                   // Read the target set filename string
         ABCDNetworkTester.targetSetFile = new File(targetSetFilename);                         // Store the target set file
      } // try
      
      catch (InputMismatchException inputMismatchException)                                     // If the scanner reads a wrong data type              
      {
         throw (new IllegalArgumentException("Invalid control file. The control file contains a mistached data type."));
      } // catch (InputMismatchException inputMismatchException)
      
      catch (NoSuchElementException noSuchElementException)                                     // If the scanner runs out of tokens prematurely
      {
         throw (new IllegalArgumentException("Invalid control file. The control file contains too few arguments."));
      } // catch (NoSuchElementException noSuchElementException)  
      
      finally
      {
         controlFileReader.close();                                                             // Always close the scanner
      } // finally
   
      return;
   } // private static void readControlFile(File controlFile) throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Creates the test network for the tester
    * 
    * preconditions: the stored file path for the network configuration file is set
    * postconditions: If the network configuration file path object identifies a valid network configuration file, 
    *                 the test network is created using the network configuration file.
    *                 If the given file path object does not identify an existing file, throws a FileNotFoundException.
    *                 If the given file path object identifies an improperly formatted control file, throws an IllegalArgumentException.
    *                 Print the filename used.
    */
   private static void createNetwork() throws FileNotFoundException, IllegalArgumentException
   {
      System.out.println("Using network configuration file " + ABCDNetworkTester.networkConfigurationFile.getName());
      ABCDNetworkTester.testNetwork = new ABCDNetwork(ABCDNetworkTester.networkConfigurationFile);
      
      return;
   } // private static void createNetwork() throws FileNotFoundException, IllegalArgumentException
   
   /*
    * Tests the test network by training or running
    * Uses super input file
    * 
    * preconditions: the test network is created properly, doTrainNotRun is set, and the input set and target set files are set
    * postconditions: if doTrainNotRun is true, trains the network on the current inputSet and targetSet, saving weights as appropriate to
    *                 the specified output file, and prints a comparison table along with the time elapsed during training.
    *                 if doTrainNotRun is false, runs the network on the current inputSet and prints an output table.
    *                 If the input set (or target set file if training) is nonexistent, throws a FileNotFoundException.
    *                 If the input set (or target set file if training) is improperly formatted, throws an IllegalArgumentException.
    *                 If an error is encountered during writing weights to a file, throws an IOException.
    */
   public static void testNetwork() throws FileNotFoundException, IllegalArgumentException, IOException
   {
      /*
       * If the control file signals to train the network, train the network and print a comparison table
       */
      if (ABCDNetworkTester.doTrainNotRun)
      {
         System.out.println("Training network...\n");
         
         /*
          * Extract the input AND target set arrays from their respective files
          * More efficient so the comparison table doesn't require a redundant file extraction
          * Also prevents file extraction from being counted as training time
          */
         System.out.println("Using input set file " + ABCDNetworkTester.inputSetFile.getName());
         double[][] inputSet = ABCDNetworkTester.testNetwork.extractInputSetFromSuperFile(inputSetFile);
         
         System.out.println("Using target set file " + ABCDNetworkTester.targetSetFile.getName());
         double[][] targetSet = ABCDNetworkTester.testNetwork.extractTargetSetFromFile(targetSetFile);
         
         /*
          * Start timer
          */
         long timeBeforeTraining = System.currentTimeMillis();
         
         /*
          * Train!
          */
         ABCDNetworkTester.testNetwork.trainOnSet(inputSet, targetSet);
         
         /*
          * End timer, calculate and print the elapsed time
          */
         long timeElapsed = System.currentTimeMillis() - timeBeforeTraining;
         System.out.println("\nMilliseconds elapsed during training = " + timeElapsed);
       
         /*
          * Print training performance, stopping flags, random range, training parameters, and unit configurations 
          * after training is finished
          */
         System.out.println();
         testNetwork.printTrainingPerformance();
         testNetwork.printTrainingFlags();
         testNetwork.printTrainingParameters();
         testNetwork.printNumUnits();
         testNetwork.printWeightInitialization();
         testNetwork.printAllocateForTraining();
         testNetwork.printWeightsOutputConfiguration();
         
         /*
          * Print a comparison table comparing targets and network outputs
          */
         System.out.println();
         ABCDNetworkTester.printComparisonTable(testNetwork, inputSet, targetSet);
      } // if (ABCDNetworkBPTester.doTrainNotRun)
      
      /*
       * If the control file indicates to only run the network, print an output table
       */
      else
      {
         System.out.println("Running network...\n");
         
         /*
          * Extract the input set arrays from the input and target set file
          * More efficient so the comparison table doesn't require a redundant file extraction
          */
         System.out.println("Using input set file " + ABCDNetworkTester.inputSetFile.getName());
         double[][] inputSet = ABCDNetworkTester.testNetwork.extractInputSetFromSuperFile(inputSetFile);

         System.out.println("Using target set file " + ABCDNetworkTester.targetSetFile.getName());
         double[][] targetSet = ABCDNetworkTester.testNetwork.extractTargetSetFromFile(targetSetFile);
         
         /*
          * Print network information
          */ 
         System.out.println();
         testNetwork.printNumUnits();
         testNetwork.printWeightInitialization();
         testNetwork.printAllocateForTraining();
         
         /*
          * Print an output table displaying network outputs
          */
         System.out.println();
         ABCDNetworkTester.printComparisonTable(testNetwork, inputSet, targetSet);
      } // else... if (ABCDNetworkBPTester.doTrainNotRun)
      
      return;
   } // public static void testNetwork() throws FileNotFoundException, IllegalArgumentException, IOException
   
   /*
    * Sets the control file using the command line arguments
    * 
    * parameters: args is the array of strings passed from the command line with the first string (should it exist) being the desired control filename
    * preconditions: args is allocated as a possibly empty array of strings
    * postconditions: if args is empty (i.e. no arguments were passed from the command line), uses the default control filename.
    *                 Otherwise, the first string passed from the command line is used as the filename
    *                 Print whether the default filename was used and which filename was set.
    */
   private static void setControlFileName(String[] args)
   {
      /*
       * If no command line arguments are given, use the default control file name provided by the tester class
       */
      if (args.length == 0)
      {
         ABCDNetworkTester.currentControlFilename = ABCDNetworkTester.defaultControlFilename;
         System.out.println("Using default control filename " + ABCDNetworkTester.currentControlFilename);
      } // if (args.length == 0)
      
      /*
       * Otherwise, use the first string given by the command line
       */
      else 
      {
         ABCDNetworkTester.currentControlFilename = args[0];
         System.out.println("Using provided control filename " + ABCDNetworkTester.currentControlFilename);
      } // if (args.length == ))... else
      
   } // private static void setControlFileName(String[] args)
   
   /*
    * Runs a test on an A-B-C-D network
    * 
    * parameters: args is the array of strings passed from the command line with the first string (should it exist) being the desired control filename
    * postconditions: results from network test are printed to the console. Any exceptions cause the test program to abort. 
    */
   public static void main(String[] args)
   {
      ABCDNetworkTester.setControlFileName(args);
      
      try
      {
         ABCDNetworkTester.readControlFile(new File(ABCDNetworkTester.currentControlFilename));
         ABCDNetworkTester.createNetwork();
         ABCDNetworkTester.testNetwork();
         
//         /*
//          * Extract input super set
//          */
//         double[][] inputSet = ABCDNetworkTester.testNetwork.extractInputSetFromSuperFile(inputSetFile);
//         
//         /*
//          * Print the input set
//          */
//         for (int member = 0; member < inputSet.length; ++member)
//         {
//            for (int alpha = 0; alpha < inputSet[member].length; ++alpha)
//            {
//               System.out.print(inputSet[member][alpha] + " ");
//            }
//            
//            System.out.println();
//         }
         
//         // Check if max or if assignment is better. If assignment is somewhat better (40% ish)
//         long startTime;
//         int numIterations = 1000000;
//         int numSuperIterations = 10000000;
//         int variable;
//         
//         // Max assignment
//         startTime = System.currentTimeMillis();
//         variable = 0;
//         
//         for (int j = 0 ; j < numSuperIterations; ++j)
//         {
//            for (int i = 0 ; i < numIterations; ++i)
//            {
//               variable = (i*i)%numIterations;
//                     
//               variable = Math.max(i, variable);
//            }
//         }
//         System.out.println("Max method took " + (System.currentTimeMillis() - startTime) + " milliseconds");
//         
//         // If assignment
//         startTime = System.currentTimeMillis();
//         variable = 0;
//         int storeI;
//         
//         for (int j = 0 ; j < numSuperIterations; ++j)
//         {
//            for (int i = 0 ; i < numIterations; ++i)
//            {
//               variable = (i*i)%numIterations;
//               
//               storeI = i;
//               if (storeI > variable)
//                  variable = i;
//            }
//         }
//         System.out.println("If method took " + (System.currentTimeMillis() - startTime) + " milliseconds");
         
         
         
      } // try
      
      catch (Exception exception)   // Catch and print any exceptions. Abort execution.
      {
         System.out.println("An exception has terminated execution:\n\t" + exception.getMessage());
         exception.printStackTrace();
      } // catch (Exception exception)
      
      return;
   } // public static void main(String[] args)
   
} // public class ABCDNetworkBPTester