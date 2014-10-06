package ru.spbsu.apmath.neuralnetwork;

import Jama.Matrix;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 17:34
 */
public class Lawyer {
  private Matrix weights;
  private Function activationFunction;
  private Matrix outputs;
  private int countOfNeurons;

  public Lawyer(Matrix weights, Function activationFunction) {
    this.weights = weights;
    this.activationFunction = activationFunction;
    this.countOfNeurons = weights.getRowDimension();
  }

  public void calculate(Matrix input) {
    outputs = functionOfMatrix(weights.times(input), activationFunction);
  }

  public Matrix getOutputs() {
    if (outputs == null) {
      throw new RuntimeException("Layer outputs must be calculated first!");
    }
    return outputs;
  }

  public int getCountOfNeurons() {
    return countOfNeurons;
  }

  public Matrix getWeights() {
    return weights;
  }

  private Matrix functionOfMatrix(Matrix m, Function function) {
    Matrix result = new Matrix(m.getRowDimension(), m.getColumnDimension());
    for (int i = 0; i < m.getRowDimension(); i++) {
      for (int j = 0; j < m.getColumnDimension(); j++) {
        result.set(i, j, function.call(m.get(i, j)));
      }
    }
    return result;
  }
}
