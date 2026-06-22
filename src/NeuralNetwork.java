import Activations.ActivationFunction;
import Activations.Sigmoid;

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
        Sigmoid sigmoid = new Sigmoid();

        for (int i = 0; i < this.layers.length; i++) {
            if (i == this.layers.length - 1){
                this.layers[i] = new Layer(layerSizes[i], layerSizes[i+1], sigmoid, LayerType.OUTPUT);
            }
            else {
                this.layers[i] = new Layer(layerSizes[i], layerSizes[i+1], sigmoid, LayerType.HIDDEN);
            }
        }
        this.learningRate = learningRate;
    }

    /**
     * percorre layers[] da esquerda para a direita.
     * Começa com o array de pixels como input da primeira camada, chama layers[0].forward(input),
     * e usa o array de outputs devolvido como input de layers[1].forward(...), e assim sucessivamente.
     * @param input array de inputs
     * @return array de outputs da última camada
     */
    private double[] forward(double[] input){
        double[] temp = new double[0];

        for (int i = 0; i < this.layers.length; i++) {
            if (i == 0){
                temp = this.layers[i].forward(input);
            }
            else{
                temp = this.layers[i].forward(temp);
            }
        }

        return temp;
    }

    /**
     * este é o mét0do mais delicado.
     * Começa por calcular os deltas da camada de output:
     * para cada neurónio i da última camada, o delta é simplesmente output[i] - target[i].
     * Passas esses deltas a layers[última].backward(deltas), que devolve um errorSignal[].
     * Esse errorSignal torna-se os deltas da camada anterior, e assim de seguida até chegares à primeira camada.
     * No fim chamas updateWeights em todas as camadas.
     * @param target
     */
    private void backward(double[] target){
        double[] deltas = new double[this.layers[this.layers.length-1].getSize()];
        double[] outputs = this.layers[this.layers.length-1].getOutputs();


        for (int i = 0; i < outputs.length; i++) {
            deltas[i] = outputs[i] - target[i];
        }

        double[] errorSignal = deltas;

        for (int i = this.layers.length-1; i >= 0; i--) {
            errorSignal = this.layers[i].backward(errorSignal);
        }

        for (int i = 0; i < this.layers.length; i++) {
            this.layers[i].updateWeights(this.learningRate);
        }
    }

    /**
     * o loop de treino principal.
     * Para cada epoch, percorre todos os exemplos do dataset;
     * para cada exemplo chama forward, depois backward com o target one-hot correspondente ao label.
     * Tipicamente no fim de cada epoch calculas a loss média e imprimes o progresso.
     * Mais tarde, quando introduzires mini-batches no Trainer,
     * este mét0do fica mais simples — delega o loop ao Trainer e só gere a lógica de alto nível.
     */
    private void train(double[][] data, int[] labels, int epochs){

        for (int i = 0; i < epochs; i++) {
            this.forward();
            this.backward();
        }

    }

    /**
     * faz apenas o forward pass e devolve um int — o índice do output com maior valor (o argmax do array de probabilidades).
     * É o dígito que a rede acha mais provável. Não há backward, não há update de pesos — é só inferência.
     * @param input
     * @return
     */
    private int predict(double[] input){

    }

    /**
     * calcula a Cross-Entropy Loss para um único exemplo: percorre as 10 posições e calcula -Σ(target[i] * log(output[i])).
     * Como o target é one-hot (apenas uma posição é 1, as restantes são 0),
     * na prática só uma posição contribui — é o negativo do logaritmo da probabilidade atribuída à classe correcta.
     * Um valor próximo de zero significa que a rede está confiante e certa; um valor alto significa erro.
     * Guardas este valor em lossHistory para monitorização.
     * @return
     */
    private double computeLoss(double[] output, double[] target){

    }






}
