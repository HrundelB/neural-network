package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.methods.VecOptimization;

import java.util.Random;

import static com.spbsu.commons.math.vectors.MxTools.multiply;
import static com.spbsu.commons.math.vectors.MxTools.transpose;
import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * Created by afonin.s on 21.11.2014.
 */
public class BackPropagation<Loss extends TargetFunc & FuncC1> extends WeakListenerHolderImpl<Perceptron>
        implements VecOptimization<Loss> {

  private double w;
  private final int numberOfSteps;
  private final int[] dims;
  private final Function activationFunction;

  public BackPropagation(int[] dims, Function activationFunction, double w, int numberOfSteps) {
    if (dims.length < 2) {
      throw new IllegalArgumentException("Perceptron must have at least two dims: input and output");
    }
    this.dims = dims;
    this.activationFunction = activationFunction;
    this.w = w;
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
    boolean[] flags = new boolean[learn.length()];
    resetFlags(flags);
    for (int k = 0; k < numberOfSteps; k++) {
      int n = 0;
      System.out.println("STEP: " + k);
      while(!allFlagsAreTrue(flags)) {
        n++;
        int index = new Random().nextInt(learn.length());
        flags[index] = true;

        Vec output = perceptron.trans(learn.at(index));

        Vec[] deltas = new Vec[weights.length];
        deltas[deltas.length - 1] = loss.gradient(output);
        scale(deltas[deltas.length - 1], activationFunction.vecDerivative(perceptron.getSum(deltas.length - 1)));
        for (int l = deltas.length - 2; l >= 0; l--) {
          deltas[l] = multiply(transpose(weights[l + 1]), deltas[l + 1]);
          scale(deltas[l], activationFunction.vecDerivative(perceptron.getSum(l)));
        }

        Mx[] deltaWeigths = new Mx[weights.length];
        for (int l = 0; l < deltaWeigths.length; l++) {
          deltaWeigths[l] = scale(outer(deltas[l], perceptron.getOutput(l - 1)), w);
        }

        for (int l = 0; l < weights.length; l++) {
          append(weights[l], deltaWeigths[l]);
        }

        perceptron = new Perceptron(weights, activationFunction);
      }
      System.out.println(n + " ITERATIONS");
      resetFlags(flags);

      invoke(perceptron);
      w = w / 1.07;
    }
    return perceptron;
  }

  private void resetFlags(boolean[] flags) {
    for (int i = 0; i < flags.length; i++) {
      flags[i] = false;
    }
  }

  private boolean allFlagsAreTrue(boolean[] flags) {
    for (int i = 0; i < flags.length; i++) {
      if (!flags[i]) {
        return false;
      }
    }
    return true;
  }

  private void fillWithRandom(Mx mx) {
    Random random = new Random();
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, random.nextDouble() / 10);
      }
    }
  }
}
