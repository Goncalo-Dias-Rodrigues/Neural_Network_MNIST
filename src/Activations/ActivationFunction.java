package Activations;

public interface ActivationFunction {

    /**
     * É chamado pelo Neuron no forward pass sobre o preActivation.
     *
     * @param x valor
     * @return valor trasnformado
     */
    public double activate(double x);

    /**
     * É chamado pelo Neuron dentro de computeDelta para escalar o sinal de erro.
     * Sem esta derivada o backpropagation não consegue saber "quão sensível" foi a ativação a uma mudança no input.
     *
     * @param x valor
     * @return valor trasnformado
     */
    public double derivative(double x);

    /**
     * versão vectorial, que recebe e devolve um array inteiro.
     * A maioria das funções aplica activate elemento a elemento,
     * mas o Softmax precisa de ver todos os valores ao mesmo tempo para normalizar
     * por isso este mét0do existe na interface.
     * Para ReLU e Sigmoid podes implementá-lo como um simples loop sobre activate.
     * @param x array
     * @return array transformado
     */
    public double[] activateArray(double[] x);
}
