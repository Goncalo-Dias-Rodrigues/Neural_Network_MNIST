

public class NeuralNetwork {
    /*
        + train(data, labels, epochs): void
        + computeLoss(out, target): double
     */

    /**
     * um array de Layer que representa a arquitectura completa da rede.
     * A ordem é fundamental: layers[0] é a primeira camada escondida,
     * layers[layers.length - 1] é sempre a camada de output.
     * No forward pass percorres o array da esquerda para a direita;
     * no backward percorres da direita para a esquerda.
     * Para o MNIST uma boa configuração de início seria duas entradas neste array:
     * uma Layer com 128 neurónios e ReLU, e uma Layer com 10 neurónios e Softmax.
     */
    private Layer[] layers;

    /**
     * um double que controla o tamanho do passo em cada update de pesos.
     * É passado a cada Layer.updateWeights() no fim de cada batch.
     * Valores típicos para o MNIST ficam entre 0.001 e 0.01.
     * Se for demasiado grande a rede diverge;
     * se for demasiado pequeno aprende muito devagar.
     */
    private double learningRate;

    /**
     * uma lista (ou array dinâmico) de double onde guardas o valor médio da loss ao fim de cada epoch.
     * Não é necessário para o treino funcionar, mas é essencial para perceberes se a rede está a aprender
     * se a loss não descer de epoch para epoch, há um problema.
     */
    private double[] lossHistory;


    /**
     * Com base nesse array crias cada Layer, e para cada camada determinas quantos inputs cada neurónio vai receber
     * que é o tamanho da camada anterior. Delegar a criação dos pesos ao WeightInitializer é a decisão certa aqui:
     * para cada camada, chamás o inicializador com (tamanho_camada_anterior, tamanho_camada_actual) e
     * passas os pesos resultantes a cada Neuron.
     *
     * layerSizes tem o formato {784, 128, 10}. O índice 0 é o número de inputs brutos (os pixels), não uma camada.
     * Crias layers com tamanho layerSizes.length - 1.
     * Para cada camada i, passas layerSizes[i] como inputSize e layerSizes[i+1] como neuronCount ao construtor da Layer.
     * A última camada recebe OUTPUT como layerType, as restantes recebem HIDDEN.
     *
     * @param layerSizes array de inteiros onde cada posição indica o número de neurónios nessa camada
     * @param learningRate racio de aprendizagem
     */
    public NeuralNetwork(int[] layerSizes, double learningRate) {
        this.layers = new Layer[layerSizes.length - 1];

        for (int i = 0; i < this.layers.length; i++) {
            if (i == this.layers.length - 1){
                this.layers[i] = new Layer(layerSizes[i], layerSizes[i+1], ActivationFunction, LayerType.OUTPUT);
            }
            else {
                this.layers[i] = new Layer(layerSizes[i], layerSizes[i+1], ActivationFunction, LayerType.HIDDEN);
            }
        }
        this.learningRate = learningRate;
    }

    private double[] forward(double[] input){

    }

    private void backward(double[] target){

    }

    private void train(){

    }

    private int predict(double[] input){

    }

    private double computeLoss()






}
