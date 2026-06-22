package Activations;

public class Sigmoid implements ActivationFunction{

    /**
     * Comprime qualquer valor real para o intervalo (0, 1).
     * @param x valor
     * @return 1.0 / (1.0 + Math.exp(-x)).
     */
    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Podes chamar activate(x) internamente e usar o resultado para calcular a derivada, evitando repetir a fórmula.
     * @param x valor
     * @return sigmoid(x) * (1 - sigmoid(x)).
     */
    @Override
    public double derivative(double x) {
        return activate(x) * (1 - activate(x));
    }

    @Override
    public double[] activateArray(double[] x) {
        double[] result = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            result[i] = this.activate(x[i]);
        }

        return result;
    }
}
