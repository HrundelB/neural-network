package ru.spbsu.apmath.neuralnetwork;


import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

import static com.spbsu.commons.math.vectors.MxTools.multiply;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 17:34
 */
public class Layer {
  private Mx weights;
  private Function activationFunction;
  private Vec outputs;
  private int countOfNeurons;

  public Layer(Mx weights, Function activationFunction) {
    this.weights = weights;
    this.activationFunction = activationFunction;
    this.countOfNeurons = weights.rows();
  }

  public void calculate(Vec input) {
    outputs = functionOfVector(multiply(weights, input), activationFunction);
  }

  public Vec getOutputs() {
    if (outputs == null) {
      throw new RuntimeException("Layer outputs must be calculated first!");
    }
    return outputs;
  }

  public int getCountOfNeurons() {
    return countOfNeurons;
  }

  public Mx getWeights() {
    return weights;
  }

  private Vec functionOfVector(Vec m, Function function) {
    Vec result = new ArrayVec(m.dim());
    for (int i = 0; i < m.dim(); i++) {
        result.set(i, function.call(m.get(i)));
    }
    return result;
  }
}
