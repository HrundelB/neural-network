package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.methods.VecOptimization;

import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * Created by afonin.s on 21.11.2014.
 */
public class BackPropagation<Loss extends Logit> extends WeakListenerHolderImpl<Perceptron>
        implements VecOptimization<Loss> {

  private final int numberOfSteps;
  private final int[] dims;
  private final FunctionC1 activationFunction;

  public BackPropagation(int[] dims, FunctionC1 activationFunction, int numberOfSteps) {
    if (dims.length < 2) {
      throw new IllegalArgumentException("Perceptron must have at least two dims: input and output");
    }
    this.dims = dims;
    this.activationFunction = activationFunction;
    this.numberOfSteps = numberOfSteps;
  }

  @Override
  public Perceptron fit(VecDataSet learn, Loss loss) {
    Mx[] weights = new Mx[dims.length - 1];
    for (int l = 0; l < dims.length - 1; l++) {
      weights[l] = new VecBasedMx(dims[l + 1], dims[l]);
      fillWithRandom(weights[l]);
    }
    Perceptron perceptron = new Perceptron(weights, activationFunction);
    for (int k = 0; k < numberOfSteps; k++) {
      perceptron = step(learn, loss, perceptron);

      invoke(perceptron);
    }
    return perceptron;
  }

  private Perceptron step(VecDataSet learn, Loss loss, Perceptron perceptron) {
    for (int t = 0; t < learn.length(); t++) {
      int index = new Random().nextInt(learn.length());

      Perceptron tmpPerceptron = perceptron.clone();
      for (int i = 0; i < tmpPerceptron.depth(); i++) {
        Mx mx = tmpPerceptron.weights(i);
        for (int j = 0; j < mx.rows(); j++) {
          int k = (int) Math.random() * 10;
          while (k < mx.columns()) {
            mx.row(j).set(k, 0);
            k += 0.1 * mx.columns() + Math.random() * 10;
          }
        }
      }

      final Vec learningVec = learn.at(index);
      tmpPerceptron.trans(learningVec);
      final int depth = tmpPerceptron.depth() - 1;

      Vec delta;

      delta = loss.gradient(tmpPerceptron.getSum(depth), index);
      append(perceptron.weights(depth), scale(outer(delta, tmpPerceptron.getOutput(depth - 1)), 0.01));

      for (int l = depth - 1; l >= 0; l--) {
        delta = MxTools.multiply(MxTools.transpose(tmpPerceptron.weights(l + 1)), delta);
        scale(delta, function.vecValue(tmpPerceptron.getSum(l)));
        append(perceptron.weights(l), scale(outer(delta, tmpPerceptron.getOutput(l - 1)), 0.01));
      }
    }
    return perceptron;
  }

  private void fillWithRandom(Mx mx) {
    final Random random = new FastRandom();
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, random.nextGaussian());
      }
    }
  }

  private Function function = new Function() {
    @Override
    public double call(double x) {
      return Math.exp(x) / Math.pow(1 + Math.exp(x), 2);
    }
  };
}
