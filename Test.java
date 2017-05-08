import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Test {
    private static final String FILEPATH = "src\\optdigits_train.txt";
    private static final String TEST_FILEPATH = "src\\optdigits_test.txt";
    static Network network;
    public static void main(String[] args){
        //first create a network with specified # of hidden layers and # of neurons per layer
        network = new Network(1, 17);

        //now we create an input layer with neurons values from file
        Layer inputLayer = new Layer();

        //create an output layer
        Layer outputLayer = new Layer();
        for(int i = 0; i < 10; i++){
            Neuron neuron = new Neuron();
            neuron.setValue(i);
            neuron.setOutputSum(i);
            outputLayer.addNeuron(neuron);
        }
        network.connectOutputLayer(outputLayer);

        int trainNumber = 120;
        //at first the setInputLayer is true. this value will be used to determine if to connect the input layer or change the input layer because connect input layer randomizes the weights
        //and change input layer keeps same weights
        System.out.println("Training...");
        boolean setInputLayer = true;
        while(trainNumber > 0) {
            //this try catch is to read the file and populate the input layer and train the network
            try {
                BufferedReader bufferedReader = new BufferedReader(new FileReader(FILEPATH));
                String content;
                while ((content = bufferedReader.readLine()) != null) {
                    String[] arrayOfIndividualNumbers = content.split(",");
                    double target = 0;
                    //if we need to add the input layer we do it here other wise we change the values
                    if (setInputLayer) {
                        for (int i = 0; i < arrayOfIndividualNumbers.length; i++) {
                            if (i == arrayOfIndividualNumbers.length - 1) {
                                target = Double.parseDouble(arrayOfIndividualNumbers[i]);
                            } else {
                                int num = Integer.parseInt(arrayOfIndividualNumbers[i]);
                                double num_norm = (double) num/16;
                                Neuron neuron = new Neuron();
                                neuron.setValue(num_norm);
                                neuron.setOutputSum(num_norm);
                                inputLayer.addNeuron(neuron);
                            }
                        }
                        network.connectInputLayer(inputLayer);
                        setInputLayer = false;
                    } else {
                        for (int i = 0; i < arrayOfIndividualNumbers.length; i++) {
                            if (i == arrayOfIndividualNumbers.length - 1) {
                                target = Double.parseDouble(arrayOfIndividualNumbers[i]);
                            } else {
                                int num = Integer.parseInt(arrayOfIndividualNumbers[i]);
                                double num_norm = (double)num/16;
                                network.getLayers().get(0).getNeurons().get(i).setValue(num_norm);
                                network.getLayers().get(0).getNeurons().get(i).setOutputSum(num_norm);
                            }
                        }
                    }
                    network.forwardPropagate();
                    network.backPropagation(target);
                }
                bufferedReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            trainNumber--;
        }

        System.out.println("Testing...");
        //test network
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(TEST_FILEPATH));
            String content;
            int countCorrect = 0;
            int countOfTries = 0;
            while ((content = bufferedReader.readLine()) != null) {
                String[]  arrayOfIndividualNumbers = content.split(",");
                int target =0;
                for (int i = 0; i < arrayOfIndividualNumbers.length; i++) {
                    if (i == arrayOfIndividualNumbers.length - 1) {
                        target = Integer.parseInt(arrayOfIndividualNumbers[i]);
                    } else {
                        int num = Integer.parseInt(arrayOfIndividualNumbers[i]);
                        double num_norm = (double) num/16;
                        network.getLayers().get(0).getNeurons().get(i).setValue(num_norm);
                        network.getLayers().get(0).getNeurons().get(i).setOutputSum(num_norm);
                    }
                }
                network.forwardPropagate();
                double guess = -1;
                double max = -1000000000;
                for(int i = 0; i < network.getLayers().get(network.getLayers().size()-1).getNeurons().size(); i++){
                    if(network.getLayers().get(network.getLayers().size()-1).getNeurons().get(i).getValue() > max){
                        max = network.getLayers().get(network.getLayers().size()-1).getNeurons().get(i).getValue();
                        guess = i;
                    }
                }
                countOfTries++;
                if(guess == target){
                    countCorrect++;
                }else{
                        //System.out.println("Did not recognize " + target);

                }


            }
            double countC = countCorrect;
            double tries = countOfTries;
            System.out.println("Correct number: " + countC);
            System.out.println("count of tries: " + tries);
            double p = (countC/tries)*100;
            System.out.println("Percent Accuracy: " + p);
            bufferedReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        //personal numbers test
        int[] one =
                {       0, 0, 0,   0,  0, 0,  0, 0,
                        0, 0, 16, 16, 16, 16, 0, 0,
                        0, 0, 16,  0, 0,  16, 0, 0,
                        0, 0, 16,  0, 0,  16, 0, 0,
                        0, 0, 16,  0, 0,  16, 0, 0,
                        0, 0, 16,  0, 0,  16, 0, 0,
                        0, 0, 16, 16, 16, 16, 0, 0,
                        0, 0, 0,   0, 0,  0,  0, 0
                };
        int[] two =
                {       0, 0, 0,   0,  0, 0,  0, 0,
                        0, 0, 16, 16, 16, 16, 0, 0,
                        0, 0, 0,   0, 0,  16, 0, 0,
                        0, 0, 0,   0, 0,  16, 0, 0,
                        0, 0, 16,  16, 16,16, 0, 0,
                        0, 0, 16,  0, 0,  0,  0, 0,
                        0, 0, 9,  16, 1,  2,  0, 0,
                        0, 0, 0,  10, 16, 16, 0, 0
                };
        int[] three =
                {       0, 0, 0,   0,  0, 0,  0, 0,
                        0, 0, 16, 16, 16, 16, 12, 0,
                        0, 0, 0,   0, 0,  16, 10, 0,
                        0, 0, 0, 16, 16,  16, 3, 0,
                        0, 0, 0,  0,  0,  16, 2, 0,
                        0, 0, 0,  0,  1,  16, 10, 0,
                        0, 0, 16, 16, 16,  16, 12, 0,
                        0, 0, 0,  0,  2,   0, 0, 0
                };
        int[] four =
                {       0, 0, 16,  0,  0,  16, 0, 0,
                        0, 0, 16,  0,  0,  16, 6, 0,
                        0, 0, 16,  0,  0,  16, 7, 0,
                        0, 0, 16,  9,  6,  16, 7, 0,
                        0, 0, 16,  16, 16, 16, 4, 0,
                        0, 0, 0,    0,  0, 16, 3, 0,
                        0, 0, 0,    0,  0, 16, 2, 0,
                        0, 0, 0,    0,  0, 16, 1, 0
                };
        int[] five =
                {       0, 0, 0,  0,  0,   0, 0, 0,
                        0, 0, 16, 16, 16, 16, 0, 0,
                        0, 0, 16,  0,  0,  0, 0, 0,
                        0, 0, 16,  0,  0,  0, 0, 0,
                        0, 0, 16, 16, 16, 16, 0, 0,
                        0, 0, 0,   0,  0, 16, 1, 0,
                        0, 0, 0,   16, 16,16, 0, 0,
                        0, 0, 16,   5, 6,  4, 0, 0
                };
        int[] five_another =
                {
                        0, 0, 0, 0,  0, 0, 0, 0,
                        0, 0, 0, 0, 16, 3, 0, 0,
                        0, 0, 0, 0, 16, 4, 0, 0,
                        0, 0, 0, 0, 16, 3, 0, 0,
                        0, 0, 0, 0, 16, 3, 0, 0,
                        0, 0, 0, 0, 16, 5, 0, 0,
                        0, 0, 3, 16, 16, 16, 3, 0,
                        0, 0, 0, 0,  0, 0, 0, 0,
                };

        int[] number =
                {
                        0, 0,  0,  0,  0,  0, 0, 0,
                        0, 0,  0, 16, 16,  0, 0, 0,
                        0, 0, 16,  0,  0, 16, 0, 0,
                        0, 0, 16, 16, 16, 16, 0, 0,
                        0, 0,  0, 16, 16,  0, 0, 0,
                        0, 0, 16, 10, 10, 16, 0, 0,
                        0, 0, 16,  0,  0, 16, 0, 0,
                        0, 0,  0, 16, 16,  0, 0, 0
                };

        for (int i = 0; i < three.length; i++){
            double num_norm = (double) three[i]/16;
            network.getLayers().get(0).getNeurons().get(i).setValue(num_norm);
            network.getLayers().get(0).getNeurons().get(i).setOutputSum(num_norm);
        }
        network.forwardPropagate();
        double guess = -1;
        double max = -1000000000;
        for(int i = 0; i < network.getLayers().get(network.getLayers().size()-1).getNeurons().size(); i++){
            if(network.getLayers().get(network.getLayers().size()-1).getNeurons().get(i).getValue() > max){
                max = network.getLayers().get(network.getLayers().size()-1).getNeurons().get(i).getValue();
                guess = i;
            }
        }
        System.out.println(" the number you drawn was: " + guess);

    }
}
