package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.commons.math.vectors.VecTools.multiply;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 22:23
 */
public class Perceptron {
  private int countOfLayers;
  private Layer[] layers;
  private List<Integer> hiddenLayersDims;

  public Perceptron(List<Integer> hiddenLayersDims) {
    this.hiddenLayersDims = hiddenLayersDims;
    this.countOfLayers = hiddenLayersDims.size() + 1;  //hidden layers + 1 output
    this.layers = new Layer[countOfLayers];
  }

  public double calculate(Vec input) {
    if (layers[0] == null) {
      throw new RuntimeException("Perceptron must be learned before work!");
    }
    layers[0].calculate(input);
    for (int i = 1; i < countOfLayers; i++)
      layers[i].calculate(layers[i-1].getOutputs());
    return layers[countOfLayers - 1].getOutputs().get(0);
  }

  public void backPropagation(double n, Vec[] learning, int[] answers, int numberOfSteps) {
    int inputDim = learning[0].dim();
    int countOfLearningSamples = learning.length;
    initWeights(inputDim);
    System.out.println("Learning...");
    for (int k = 0; k < numberOfSteps; k++) {
      for (int d = 0; d < countOfLearningSamples; d++){
        this.calculate(learning[d]);
        Vec[] deltas = new Vec[countOfLayers];
        deltas[countOfLayers - 1] = getDeltaForLastLayer(answers[d]);
        for (int l = countOfLayers - 2; l >= 0; l--) {
          deltas[l] = getDeltaForLayer(l, deltas[l + 1]);
        }
        changeWeights(n, learning[d], deltas);
      }
    }
  }

  private void changeWeights(double n, Vec learningVec, Vec[] deltas) {
    List<Vec> outputs = new ArrayList<Vec>();
    outputs.add(learningVec);
    for (int i = 0; i < countOfLayers; i++) {
      outputs.add(layers[i].getOutputs());
    }
    for (int l = 0; l < countOfLayers; l++) {
      for (int i = 0; i < layers[l].getWeights().rows(); i++) {
        for (int j = 0; j < layers[l].getWeights().columns(); j++) {
          layers[l].getWeights().adjust(i, j, n * deltas[l].get(i) * outputs.get(l).get(j));
        }
      }
    }
  }

  private Vec getDeltaForLastLayer(int answer) {
    double result;
    double o = layers[countOfLayers - 1].getOutputs().get(0);
    if (answer == 1) {
      System.out.println("Log likelihood function: " + Math.log(o));
      result = 1 - o;
    } else {
      if (answer == -1) {
        System.out.println("Log likelihood function: " + Math.log(1 - o));
        result = (o - o * o)/(o - 1);
      } else {
        throw new IllegalArgumentException("Answer must be only 1 or -1");
      }
    }
    return new ArrayVec(result);
  }

  private Vec getDeltaForLayer(int l, Vec previousDelta) {
    int neurons = layers[l].getCountOfNeurons();
    VecBuilder vecBuilder = new VecBuilder(neurons);
    for (int j = 0; j < neurons; j++) {
      double o = layers[l].getOutputs().get(j);
      Mx weights = layers[l + 1].getWeights();
      Vec vec = new ArrayVec(weights.col(j).toArray());
      vecBuilder.append(o * (1 - o) * multiply(vec, previousDelta));
    }
    return vecBuilder.build();
  }

  private void initWeights(int inputDim) {
    List<Integer> dims = new ArrayList<Integer>();
    dims.add(inputDim);
    dims.addAll(hiddenLayersDims.subList(0, hiddenLayersDims.size()));
    dims.add(1);
    for (int l = 0; l < countOfLayers; l++) {
      Vec[] rows = new Vec[dims.get(l + 1)];
      for (int i = 0; i < dims.get(l + 1); i++) {
        VecBuilder vecBuilder = new VecBuilder(dims.get(l));
        for (int j = 0; j < dims.get(l); j++) {
          vecBuilder.append(Math.random());
        }
        rows[i] = vecBuilder.build();
      }
      Mx w = new RowsVecArrayMx(rows);
      layers[l] = new Layer(w, getActivateFunction());
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
