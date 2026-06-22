import java.util.Arrays;

public class Neuron {

    /**
     * um array de double com tamanho igual ao número de inputs que este neurónio recebe.
     * Cada posição i contém o peso que multiplica o input i.
     * São estes os valores que a rede aprende ao longo do treino.
     * No início precisam de ser inicializados com valores pequenos e aleatórios.
     */
    private double[] weights;

    /**
     * um único double que é somado ao resultado da soma ponderada.
     * O bias permite ao neurónio "deslocar" a sua função de ativação,
     * dando-lhe a capacidade de activar mesmo quando todos os inputs são zero.
     * É tratado como um peso extra cujo input é sempre 1.
     */
    private double bias;

    /**
     * um double que guarda o resultado depois de aplicar a função de ativação ao preActivation.
     * É este o valor que este neurónio envia para os neurónios da camada seguinte.
     */
    private double output;

    /**
     * um double que representa o "quão responsável" este neurónio é pelo erro total.
     * É calculado durante o backward pass.
     * Para um neurónio de output, o delta vem diretamente do erro (diferença entre o output e o target).
     * Para um neurónio de camada escondida, vem da soma dos deltas da camada seguinte,
     * pesados pelos pesos correspondentes.
     */
    private double delta;

    /**
     * um array de double com o mesmo tamanho que weights.
     * Em cada posição i acumulas o gradiente do peso i, calculado como delta * inputs[i].
     * O motivo de acumular (e não actualizar imediatamente) é para suportares mini-batches
     * somas os gradientes de vários exemplos antes de fazer um único update.
     */
    private double[] gradWeights;

    /**
     * um double que guarda o resultado da soma ponderada antes de passar pela função de ativação,
     * ou seja, o valor Σ(wᵢ · xᵢ) + bias. Tens de o guardar porque precisas dele no backward pass:
     * para calcular a derivada da função de ativação, precisas do valor que entrou nela.
     */
    private double preActivation;

    /**
     * um double onde acumulas o gradiente do bias.
     * É simplesmente igual ao delta do neurónio
     * (porque o input "implícito" do bias é sempre 1, então delta * 1 = delta).
     */
    private double gradBias;

    /**
     * Dentro do construtor atribuis this.weights = weights, this.bias = bias,
     * e inicializas gradWeights como um array de zeros com o mesmo tamanho que weights, e gradBias a zero.
     * Os campos preActivation, output e delta ficam a zero por defeito do Java.
     * @param weights pesos já calculados externamente (pelo WeightInitializer).
     * @param bias bias inicial (normalmente zero).
     */
    public Neuron(double[] weights, double bias) {
        this.weights = weights.clone();
        this.bias = bias;
        this.output = 0;
        this.delta = 0;
        this.gradWeights = new double[this.weights.length];
        this.gradBias = 0;
    }

    /**
     * Dentro deste mét0do tens de fazer três coisas em sequência:
     * calcular a soma ponderada Σ(weights[i] * inputs[i]),
     * somar o bias,
     * guardar o resultado em preActivation, e depois passar esse valor pela função de ativação,
     * guardando o resultado em output. O mét0do devolve o output.
     * @param inputs array de outputs da camada anterior (ou os pixels da imagem, se for a primeira camada).
     * @return resultado da função de ativação.
     */
    public double forward(double[] inputs){
         double sum = 0;

        for (int i = 0; i < inputs.length; i++) {
            sum += this.weights[i] * inputs[i];
        }

        sum += this.bias;

        preActivation = sum;

        this.output = ActivationFunction.activate(preActivation);

        return this.output;
    }

    /**
     * Para um neurónio de output é a diferença (output - target).
     * Para um neurónio escondido é a soma ponderada dos deltas da camada à frente.
     * Dentro do mét0do multiplicas esse errorSignal pela derivada da função de ativação avaliada em preActivation.
     * Esse produto é o delta deste neurónio. É este valor que vai "propagar para trás" para a camada anterior.
     * @param errorSignal double que representa o sinal de erro vindo de fora.
     */
    public void computeDelta(double errorSignal){
        this.delta = errorSignal * ActivationFunction.derivative(preActivation);
    }

    /**
     *  Para cada peso i, adicionas delta * inputs[i] ao gradWeights[i]. Adicionas também delta ao gradBias.
     *  Este mét0do é chamado depois de computeDelta e antes de updateWeights.
     * @param inputs os mesmos inputs que foram usados no forward pass e acumula os gradientes.
     */
    public void accumulateGradients(double[] inputs){
        for (int i = 0; i < inputs.length; i++) {
            this.gradWeights[i] += (this.delta * inputs[i]);
        }
        this.gradBias += delta;
    }

    /**
     * Aplica a regra do gradient descent: para cada peso i, subtrais learningRate * gradWeights[i] ao weights[i].
     * Faz o mesmo para o bias.
     * No final deste mét0do tens de repor gradWeights a zeros e gradBias a zero
     * para que a próxima acumulação comece do zero.
     * @param learningRate racio de aprendizagem
     */
    public void updateWeights(double learningRate){
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] -= learningRate * this.gradWeights[i];
        }
        this.bias -= learningRate * this.gradBias;

        Arrays.fill(this.gradWeights, 0);
        this.gradBias = 0;
    }

    /**
     * Necessário para o ModelSerializer guardar o estado da rede, e para a camada
     * anterior calcular o sinal de erro que lhe compete.
     * @return uma cópia do array weights.
     */
    public double[] getWeights(){
        return this.weights.clone();
    }

    public double getDelta() {
        return delta;
    }

    public double getOutput() {
        return output;
    }
}

