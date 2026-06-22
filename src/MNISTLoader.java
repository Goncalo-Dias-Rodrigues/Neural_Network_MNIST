import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public class MNISTLoader {
    /*
        + normalize(img): double[]
        + oneHot(label): double[]
     */

    private String imagesPath;
    private String labelsPath;

    public MNISTLoader(String imagesPath, String labelsPath) {
        this.imagesPath = imagesPath;
        this.labelsPath = labelsPath;
    }

    /**
     *  abre o ficheiro de imagens com um DataInputStream.
     *  Lê e descarta os 16 bytes do cabeçalho com quatro chamadas a readInt().
     *  Depois, para cada imagem, lê 784 bytes com readUnsignedByte() (importante usar a versão unsigned
     *  readByte devolve valores entre -128 e 127, que depois da normalização dão valores negativos e estragam o treino).
     *  Guarda cada imagem como um double[] de tamanho 784 num array bidimensional double[][]. Devolve esse array.
     * @return array de imagens, cada uma com os pixeis entre 0 e 1
     */
    private double[][] loadImages(){
        int imageSize = 784;
        double[][] images = null;

        try (DataInputStream dis = new DataInputStream(new FileInputStream(this.imagesPath))) {

            dis.readInt();
            int numberOfImages = dis.readInt();
            System.out.println("A carregar " + numberOfImages + " imagens...");

            //Lê e descarta o número de linhas e colunas
            dis.readInt();
            dis.readInt();

            //Inicializa o array com o tamanho correto lido do ficheiro
            images = new double[numberOfImages][imageSize];

            for (int i = 0; i < numberOfImages; i++) {
                for (int j = 0; j < imageSize; j++) {
                    images[i][j] = dis.readUnsignedByte();
                }
                images[i] = this.normalize(images[i]);
            }

        } catch (IOException e) {
            System.err.println("Erro ao ler o ficheiro de imagens: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
        return images;
    }

    /**
     * abre o ficheiro de labels com um DataInputStream.
     * Lê e descarta 8 bytes do cabeçalho.
     * Depois lê um byte por label com readUnsignedByte() e guarda num array int[]. Devolve esse array.
     * @return array de labels
     */
    private int[] loadLabels(){
        int[] labels = null;

        try (DataInputStream dis = new DataInputStream(new FileInputStream(this.labelsPath))) {

            dis.readInt();
            int numberOfLabels = dis.readInt();
            System.out.println("A carregar " + numberOfLabels + " labels...");
            dis.readInt();
            dis.readInt();

            labels = new int[numberOfLabels];

            for (int i = 0; i < numberOfLabels; i++) {
                labels[i] = dis.readUnsignedByte();
            }

        } catch (IOException e) {
            System.err.println("Erro ao ler o ficheiro de labels: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
        return labels;

    }

    /**
     * recebe um array de pixels com valores entre 0.0 e 255.0 e divide cada um por 255.0,
     * produzindo valores entre 0.0 e 1.0.
     * Redes neuronais são sensíveis à escala dos inputs — sem normalização os gradientes explodem ou desaparecem.
     * Podes chamar este mét0do dentro de loadImages() directamente em vez de expô-lo separadamente,
     * mas tê-lo isolado facilita testes.
     * @param image array de pixels com valores entre 0.0 e 255.0
     * @return array de pixels com valores entre 0.0 e 1.0
     */
    private double[] normalize(double[] image){
        double[] result = new double[image.length];
        for (int i = 0; i < image.length; i++) {
            result[i] = image[i] / 255.0;
        }
        return result;
    }

    /**
     *  É chamado durante o loop de treino para cada exemplo, não antecipadamente para todos.
     * @return array de double com tamanho 10 preenchido a zeros, com 1.0 na posição label.
     */
    private double[] oneHot(int label){
        double[] result = new double[10];
        Arrays.fill(result, 0.0);
        result[label] = 1.0;
        return result;
    }
}
