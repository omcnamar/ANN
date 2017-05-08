import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class Neuron {
    private double value;
    private double outputSum;
    private ArrayList<Double> weights;
    private ArrayList<Double> oldWeights;
    private ArrayList<Neuron> connections;

    public Neuron(){
        connections = new ArrayList<>();
        weights     = new ArrayList<>();
        oldWeights  = new ArrayList<>();
    }
    public void setConnections(ArrayList<Neuron> neurons){
        for(int i = 0; i < neurons.size(); i++){
            connections.add(neurons.get(i));
            double Low = 0;
            double High = 1;
            double random = ThreadLocalRandom.current().nextDouble(Low, High);
            random = Math.round(random*100.0)/100.0;
            weights.add(random);
            oldWeights.add(random);
        }
    }
    public ArrayList<Neuron> getConnections(){
        return connections;
    }
    public ArrayList<Double> getWeights(){
        return weights;
    }
    public ArrayList<Double> getOldWeights(){
        return oldWeights;
    }
    public void setValue(double value){
        this.value = value;
    }
    public double getValue(){
        return value;
    }
    public void setOutputSum(double outputSum){
        this.outputSum = outputSum;
    }
    public double getOutputSum(){
        return outputSum;
    }

}
