package ru.spbsu.apmath.neuralnetwork;

import Jama.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 22:23
 */
public class Perceptron {
  private int countOfLayers;
  private Lawyer[] layers;
  private List<Integer> hiddenLayersDims;

  public Perceptron(List<Integer> hiddenLayersDims) {
    this.hiddenLayersDims = hiddenLayersDims;
    this.countOfLayers = hiddenLayersDims.size() + 1;  //hidden layers + 1 output
    this.layers = new Lawyer[countOfLayers];
  }

  public Matrix calculate(Matrix input) {
    if (layers[0] == null) {
      throw new RuntimeException("Perceptron must be learned before work!");
    }
    layers[0].calculate(input);
    for (int i = 1; i < countOfLayers; i++)
      layers[i].calculate(layers[i-1].getOutputs());
    return layers[countOfLayers - 1].getOutputs();
  }

  public void backPropagation(double n, Matrix[] learning, Matrix[] answers, int numberOfSteps, Function p) {
    int inputDim = learning[0].getRowDimension();
    int outputDim = answers[0].getRowDimension();
    int countOfLearningSamples = learning.length;
    initWeights(inputDim, outputDim);
    for (int i = 0; i < numberOfSteps; i++) {
      for (int d = 0; d < countOfLearningSamples; d++){
        this.calculate(learning[d]);
        Matrix[] deltas = new Matrix[countOfLayers];
        deltas[countOfLayers - 1] = getDeltaForLastLayer(answers[d], p);
        for (int l = countOfLayers - 2; l >= 0; l--) {
          deltas[l] = getDeltaForLayer(answers[d], p, l, deltas[l+1]);
        }
        layers[0].getWeights().plus(deltas[0].times(learning[d].transpose()).times(n));
        for (int l = 1; l < countOfLayers; l++) {
          layers[l].getWeights().plus(deltas[l].times(layers[l-1].getOutputs().transpose()).times(n));
        }
      }
    }
  }

  private Matrix getDeltaForLastLayer(Matrix answer, Function p) {
    int neurons = layers[countOfLayers - 1].getCountOfNeurons();
    Matrix delta = new Matrix(neurons, 1);
    for (int j = 0; j < neurons; j++) {
      double o = layers[countOfLayers - 1].getOutputs().get(j, 1);
      double t = answer.get(j, 1);
      delta.set(j, 1, p.call(o, t) * (o - o * o));
    }
    return delta;
  }

  private Matrix getDeltaForLayer(Matrix answer, Function p, int l, Matrix previousDelta) {
    int neurons = layers[l].getCountOfNeurons();
    Matrix delta = new Matrix(neurons, 1);
    for (int j = 0; j < neurons; j++) {
      double o = layers[l].getOutputs().get(j, 1);
      double t = answer.get(j, 1);
      Matrix weights = layers[l + 1].getWeights();
      delta.set(j, 1, o * (1 - o) * weights.getMatrix(0, weights.getRowDimension(), j, j).transpose().times(previousDelta).det());
    }
    return delta;
  }

  private void initWeights(int inputDim, int outputDim) {
    List<Integer> dims = new ArrayList<Integer>();
    dims.add(inputDim);
    dims.addAll(hiddenLayersDims.subList(0, hiddenLayersDims.size()));
    dims.add(outputDim);
    for (int l = 0; l <= hiddenLayersDims.size(); l++) {
      Matrix w = new Matrix(dims.get(l + 1), dims.get(l));
      for (int i = 0; i < dims.get(l + 1); i++) {
        for (int j = 0; j < dims.get(l); j++) {
          w.set(i, j, Math.random());
        }
      }
      layers[l] = new Lawyer(w, getActivateFunction());
    }
  }

  private Function getActivateFunction() {
    return new Function() {
        @Override
        public double call(double... x) {
          return 1/(1 + Math.exp(-1 * x[0]));
        }
      };
  }
}
