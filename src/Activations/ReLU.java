package Activations;

public class ReLU implements ActivationFunction{

    /**
     * Matematicamente é max(0, x).
     * @param x valor
     * @return x se x > 0, senão devolve 0.
     */
    @Override
    public double activate(double x) {
        if (x > 0){
            return x;
        }
        return 0;
    }

    /**
     * A derivada do ReLU é uma função degrau: ou passa o gradiente inteiro, ou bloqueia-o completamente.
     * Nota que recebes o preActivation guardado no neurónio não o output
     * porque precisas de saber se o neurónio "estava activo" durante o forward pass.
     * @param x valor
     * @return 1.0 se x > 0, senão devolve 0.0.
     */
    @Override
    public double derivative(double x) {
        if (x > 0){
            return 1.0;
        }
        return 0.0;
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
