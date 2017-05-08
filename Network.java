import java.util.ArrayList;

public class Network {
    private int numberOfHiddenLayers;
    private int numberOfNeuronsPerLayer;
    private ArrayList<Layer> layers;

    //constructor
    public Network(int numberOfHiddenLayers, int numberOfNeuronsPerLayer){
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.numberOfNeuronsPerLayer = numberOfNeuronsPerLayer;
        layers = new ArrayList<>();
        createLayers();
    }
    //create all the layers with specified # of neurons
    private void createLayers(){
        //loop to create the number of hidden layers
        for(int i = 0; i < numberOfHiddenLayers; i++){
            Layer layer = new Layer();
            //loop to create number of neurons per hidden layer
            for(int j = 0; j < numberOfNeuronsPerLayer; j++){
                Neuron neuron = new Neuron();
                layer.addNeuron(neuron);
            }
            layers.add(layer);
        }
        //call the connect function to connect all layers
        connect();
    }

    //connect all neurons
    private void connect(){
        for(int i = 0; i < numberOfHiddenLayers-1; i++){
            for(int j = 0; j < numberOfNeuronsPerLayer; j++){
                layers.get(i).getNeurons().get(j).setConnections(layers.get(i+1).getNeurons());
            }
        }
    }

    //function that connects provided layer to the network as an input layer
    public void connectInputLayer(Layer layer){
        for(int i = 0; i < layer.getNeurons().size(); i++){
            layer.getNeurons().get(i).setConnections(layers.get(0).getNeurons());
        }
        layers.add(0,layer);
    }
    //function that connects provided layer to the network as an output layer
    public void connectOutputLayer(Layer layer){
        //connect the last layer in the Network to the out put layer
        for(int i = 0; i < layers.get(layers.size()-1).getNeurons().size(); i++){
            layers.get(layers.size()-1).getNeurons().get(i).setConnections(layer.getNeurons());
        }
        layers.add(layer);
    }

    //forward propagate
    public void forwardPropagate(){
        //i loops through each layer in the network starting from 1
        for(int i=1; i < layers.size(); i++){
            //j loops through each neuron in the ith layers
            for(int j=0; j < layers.get(i).getNeurons().size(); j++){
                layers.get(i).getNeurons().get(j).setValue(0);
                //k loops through the previous neurons to determine the value of the current neuron
                for(int k = 0; k < layers.get(i-1).getNeurons().size(); k++) {
                    double outputSum = (layers.get(i).getNeurons().get(j).getValue())+
                            ((layers.get(i-1).getNeurons().get(k).getValue())*(layers.get(i-1).getNeurons().get(k).getWeights().get(j)));
                    layers.get(i).getNeurons().get(j).setValue(outputSum);
                    layers.get(i).getNeurons().get(j).setOutputSum(outputSum);
                }
                //after we summed the product we use the activation function
                double valueAfterSigmoid = (1/(1+Math.pow(Math.E, (-layers.get(i).getNeurons().get(j).getValue()))));
                layers.get(i).getNeurons().get(j).setValue(valueAfterSigmoid);
            }
        }
    }

    public void backPropagation(double target){
        //loop though each layer
        for(int i = layers.size()-1; i > 0; i--){
            //loop through each neuron in the ith layer and set delta output sum
            for(int j = 0; j < layers.get(i).getNeurons().size(); j++) {

                //calculate delta output sum only on the output layer with target
                if(i == layers.size() - 1) {
                    //calculate the error of output neurons
                    //if j = target than that is where you want 1 as target otherwise target is zero
                    if(j == target) {
                        double deltaOutputSum = derivativeOfSigmoidAt(layers.get(i).getNeurons().get(j).getOutputSum()) *
                                (1 - layers.get(i).getNeurons().get(j).getValue());
                        //replace current output sum with new proposed delta output sum
                        layers.get(i).getNeurons().get(j).setOutputSum(deltaOutputSum);
                    }else{
                        double deltaOutputSum = derivativeOfSigmoidAt(layers.get(i).getNeurons().get(j).getOutputSum()) *
                                (0 - layers.get(i).getNeurons().get(j).getValue());
                        //replace current output sum with new proposed delta output sum
                        layers.get(i).getNeurons().get(j).setOutputSum(deltaOutputSum);
                    }

                }else{
                    double deltaOutputSum = derivativeOfSigmoidAt(layers.get(i).getNeurons().get(j).getOutputSum());
                    //loop through to get the error
                    layers.get(i).getNeurons().get(j).setOutputSum(0);
                    for(int q = 0; q < layers.get(i+1).getNeurons().size(); q++) {
                                layers.get(i).getNeurons().get(j).setOutputSum(layers.get(i).getNeurons().get(j).getOutputSum() +
                                        (layers.get(i + 1).getNeurons().get(q).getOutputSum() * layers.get(i).getNeurons().get(j).getOldWeights().get(q)));
                    }
                    layers.get(i).getNeurons().get(j).setOutputSum( layers.get(i).getNeurons().get(j).getOutputSum() * deltaOutputSum );
                }

            }

            //loop through each neuron on the previous layer
            for(int k = 0; k < layers.get(i-1).getNeurons().size(); k++){
                //loop through each weight
                for(int h = 0; h < layers.get(i-1).getNeurons().get(k).getWeights().size(); h++) {
                    //first remove the old weight
                    layers.get(i - 1).getNeurons().get(k).getWeights().remove(h);
                    //now add back the trained weight
                    double newWeight = (layers.get(i - 1).getNeurons().get(k).getOldWeights().get(h) +
                            (layers.get(i).getNeurons().get(h).getOutputSum() * layers.get(i - 1).getNeurons().get(k).getValue()));
                    layers.get(i - 1).getNeurons().get(k).getWeights().add(h, newWeight);
                }
            }

        }

        //update old weights for next time
        for(int i = 0; i < layers.size(); i++){
            for(int j = 0; j < layers.get(i).getNeurons().size(); j++){
                for(int h = 0; h < layers.get(i).getNeurons().get(j).getWeights().size(); h++){
                    layers.get(i).getNeurons().get(j).getOldWeights().remove(h);
                    layers.get(i).getNeurons().get(j).getOldWeights().add(h,layers.get(i).getNeurons().get(j).getWeights().get(h));
                }
            }
        }
    }

    //function S'(x) derivative of sigmoid at value x is the return
    public double derivativeOfSigmoidAt(double x){
        double derivative = (Math.pow(Math.E,x))/Math.pow((Math.pow(Math.E,x)+1),2);
        return derivative;
    }

    public ArrayList<Layer> getLayers(){
        return layers;
    }
}
