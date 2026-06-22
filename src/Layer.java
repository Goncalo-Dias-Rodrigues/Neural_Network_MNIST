

public class Layer {

    /**
     * um array de Neuron com tamanho igual ao número de neurónios desta camada.
     * É o núcleo da classe: tudo o que a Layer faz é coordenar o que cada neurónio já sabe fazer individualmente.
     * O tamanho deste array define a "largura" da camada para o MNIST uma boa escolha seria,
     * por exemplo, 128 neurónios na camada escondida e 10 na camada de output (um por dígito).
     */
    private Neuron[] neurons;

    /**
     * um enum (com valores HIDDEN e OUTPUT) que indica que tipo de camada esta é.
     * Este campo é importante porque a camada de output tem um comportamento ligeiramente diferente no backward pass
     * o seu delta é calculado directamente a partir do erro em relação ao target,
     * enquanto uma camada escondida recebe o sinal de erro da camada à sua frente.
     */
    private LayerType layerType;

    /**
     * uma referência à interface ActivationFunction.
     * A Layer delega completamente a lógica de ativação para este objecto
     * ela própria não sabe se está a usar ReLU, Sigmoid ou Softmax.
     * Isso é decidido quando a camada é construída, e é passado no construtor.
     * Tipicamente camadas escondidas usam ReLU e a camada de output usa Softmax.
     */
    private ActivationFunction activationFunction;

    /**
     * um array de double que guarda uma cópia dos inputs recebidos no forward.
     * Precisas de os guardar porque no backward vais precisar de os passar a cada neurónio para ele acumular os gradientes.
     * Sem guardar este array, terias de os passar de fora, o que complica a interface.
     */
    private double[] inputs;

    /**
     * Recebe o número de inputs que cada neurónio vai receber (inputSize — que é o tamanho da camada anterior),
     * o número de neurónios a criar, a função de ativação, e o tipo de camada.
     * Dentro do construtor crias o array neurons com tamanho neuronCount, e para cada posição instancias um Neuron
     * pedindo ao WeightInitializer um array de pesos de tamanho inputSize e passando-o ao construtor do Neuron.
     * Inicializas também o array inputs com tamanho inputSize para estar pronto a receber dados no forward pass.
     * @param inputSize numero de inputs de cada neuron
     * @param neuronCount numero de neurons neste layer
     * @param layerType tipo de layer
     * @param activationFn funcao de ativacao
     */
    public Layer(int inputSize, int neuronCount, ActivationFunction activationFn, LayerType layerType){
        this.inputs = new double[inputSize];
        this.neurons = new Neuron[neuronCount];

        for (int i = 0; i < neuronCount; i++) {
            this.neurons[i] = new Neuron(WeightInitializer.random(inputSize), 0);
        }

        this.layerType = layerType;
        this.activationFunction = activationFn;
    }

    /**
     * Dentro do mét0do tens de fazer três coisas:
     * guardar o array inputs no campo homónimo para uso posterior no backward;
     * percorrer cada Neuron em neurons[] e chamar neuron.forward(inputs), recolhendo o output de cada um num array temporário;
     * e por fim, se a função de ativação for aplicada ao nível da camada
     * (como o Softmax, que precisa de ver todos os outputs ao mesmo tempo antes de normalizar),
     * aplicá-la aqui sobre esse array completo.
     * @param inputs outputs da camada anterior (ou a imagem, se for a primeira camada).
     * @return array de outputs desta camada.
     */
    private double[] forward(double[] inputs){
        this.inputs = inputs.clone();
        double[] temp = new double[0];

        for (int i = 0; i < this.neurons.length; i++) {
            temp[i] = this.neurons[i].forward(inputs);
        }

        return temp.clone();
    }

    /**
     * Para cada neurónio i, chamas neurons[i].computeDelta(deltas[i]) e depois neurons[i].accumulateGradients(this.inputs).
     * Depois de processar todos os neurónios, este mét0do tem de calcular e devolver o errorSignal para a camada anterior
     * um array onde a posição j contém a soma de neurons[i].delta * neurons[i].weights[j] para todos os i.
     * É esta soma ponderada que "propaga o erro para trás" através das ligações entre camadas.
     * @param deltas um array de double onde cada posição i contém o sinal de erro atribuído ao neurónio i desta camada.
     * @return array de errorSignal para a camada anterior
     */
    private double[] backwards(double[] deltas){
        for (int i = 0; i < this.neurons.length; i++) {
            neurons[i].computeDelta(deltas[i]);
            neurons[i].accumulateGradients(this.inputs);
        }

        double[] errorSignal = new double[this.inputs.length];

        for (Neuron neuron : this.neurons) {
            for (int j = 0; j < this.inputs.length; j++) {
                errorSignal[j] += neuron.getDelta() * neuron.getWeights()[j];
            }
        }

        return errorSignal;
    }

    /**
     * percorre todos os neurónios e chama neuron.updateWeights(learningRate) em cada um.
     * Este mét0do é chamado pela NeuralNetwork depois de processar todos os exemplos de um batch.
     * @param learningRate racio de aprendizagem
     */
    private void updateWeights(double learningRate){
        for (Neuron neuron : this.neurons) {
            neuron.updateWeights(learningRate);
        }
    }

    /**
     * É um mét0do auxiliar usado pela NeuralNetwork para encadear camadas:
     * o output desta camada é passado como input para a seguinte.
     * @return array de double com o output actual de cada neurónio.
     */
    private double[] getOutputs(){
        double[] outputs = new double[this.neurons.length];
        for (int i = 0; i < this.neurons.length; i++) {
            outputs[i] = this.neurons[i].getOutput();
        }
        return outputs;
    }

    /**
     * Útil quando a camada seguinte precisa de saber quantas entradas vai
     * receber para inicializar os pesos dos seus neurónios.
     * @return quantidade de neurons no layer
     */
    public int getSize(){
        return this.neurons.length;
    }



}
