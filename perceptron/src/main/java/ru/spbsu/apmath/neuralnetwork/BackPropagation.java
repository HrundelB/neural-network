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
    for (int k = 0; k < 100 * numberOfSteps; k++) {
      for (int t = 0; t < learn.length(); t++) {
        int index = new Random().nextInt(learn.length());

        final Vec learningVec = learn.at(index);
        perceptron.trans(learningVec);
        final int depth = perceptron.depth() - 1;

        final Vec[] deltas = new Vec[depth + 1];

        deltas[depth] = loss.gradient(perceptron.getSum(depth), index);
        append(perceptron.weights(depth), scale(outer(deltas[depth], perceptron.getOutput(depth - 1)), 0.001));

        for (int l = depth - 1; l >= 0; l--) {
          deltas[l] = function.vecValue(perceptron.getSum(l));
          scale(deltas[l], MxTools.multiply(MxTools.transpose(perceptron.weights(l + 1)), deltas[l + 1]));
          append(perceptron.weights(l), scale(outer(deltas[l], perceptron.getOutput(l - 1)), 0.001));
        }

        //              System.out.println(String.format("k:%s, t:%s, l:%s, j:%s, depth:%s, currentWeights.col(j):%s, deltas[l+1]:%s",
//                      k, t, l, j, depth, currentWeights.col(j), deltas[l + 1]));
//
//        final Mx[] deltaWeigths = new Mx[weights.length];
//
//
//        Vec delta = loss.gradient(output);
//        for (int l = weights.length - 1; l >= 0; l--) {
//          scale(delta, activationFunction.vecDerivative(perceptron.getSum(l)));
//          deltaWeigths[l] = outer(delta, perceptron.getOutput(l - 1));
//          delta = multiply(transpose(weights[l]), delta);
//        }
//
//        Function function = new Function() {
//          @Override
//          public double call(double x) {
//            Mx[] tmpMxes = new Mx[deltaWeigths.length];
//            for (int l = 0; l < deltaWeigths.length; l++) {
//              tmpMxes[l] = new VecBasedMx(deltaWeigths[l]);
//              scale(tmpMxes[l], x);
//            }
//            return new Perceptron(tmpMxes, activationFunction).trans(learningVec).get(0);
//          }
//        };
//        double w = GoldenSectionSearch.searchMax(0, 1, function, 0.00000001);
//
//        for (int l = 0; l < weights.length; l++) {
//          scale(deltaWeigths[l], w);
//          append(weights[l], deltaWeigths[l]);
//        }
//
//        perceptron = new Perceptron(weights, activationFunction);
      }

      invoke(perceptron);
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
