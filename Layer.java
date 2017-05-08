import java.util.ArrayList;

public class Layer {
    private ArrayList<Neuron> neurons;

    public Layer(){
        neurons = new ArrayList<>();
    }
    public void addNeuron(Neuron neuron){
        neurons.add(neuron);
    }
    public ArrayList<Neuron> getNeurons(){
        return neurons;
    }
}
