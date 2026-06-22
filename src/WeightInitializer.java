import java.util.Random;

public class WeightInitializer {

    private final Random random = new Random();
    /**
     *  gera pesos aleatórios dentro do intervalo [-limite, +limite] onde limite = Math.sqrt(6.0 / (inputSize + outputSize)).
     *  Esta inicialização foi desenhada para manter a variância dos sinais estável ao longo das camadas,
     *  e funciona bem com Sigmoid e Tanh. Usas Math.random() (que devolve [0,1)) e
     *  transformas para o intervalo correcto com (Math.random() * 2 - 1) * limite.
     * @param inputSize numero de inputs
     * @param outputSize numero de neuronios
     * @return array de pesos aleatorios
     */
    private double[][] xavier(int inputSize, int outputSize){
        double[][] result = new double[outputSize][inputSize];
        double limite = Math.sqrt(6.0 / (inputSize + outputSize));

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                result[i][j] = (this.random.nextDouble(0,1) * 2 - 1) * limite;
            }
        }

        return result;
    }

    /**
     *  semelhante ao Xavier mas desenhado especificamente para ReLU.
     *  O limite é calculado como Math.sqrt(2.0 / inputSize) e os pesos são gerados com uma distribuição gaussiana
     *  em Java podes usar new Random().nextGaussian() * Math.sqrt(2.0 / inputSize).
     *  É a inicialização recomendada para as camadas escondidas do teu MNIST porque vais usar ReLU nelas.
     * @param inputSize numero de inputs
     * @param outputSize numero de neuronios
     * @return array de pesos aleatorios
     */
    private double[][] he(int inputSize, int outputSize){
        double[][] result = new double[outputSize][inputSize];
        double limite = Math.sqrt(2.0 / inputSize);

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                result[i][j] = this.random.nextGaussian() * limite;
            }
        }

        return result;

    }

    /**
     * a versão mais simples: valores aleatórios pequenos entre -0.01 e 0.01.
     * Funciona mas converge mais lentamente que He ou Xavier.
     * Útil para testar que a estrutura da rede está certa antes de te preocupares com a velocidade de treino.
     * @param inputSize numero de inputs
     * @param outputSize numero de neuronios
     * @return array de pesos aleatorios
     */
    public double[][] randomW(int inputSize, int outputSize){
        double[][] result = new double[outputSize][inputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                result[i][j] = this.random.nextDouble(-0.01, 0.01);
            }
        }

        return result;
    }

}
