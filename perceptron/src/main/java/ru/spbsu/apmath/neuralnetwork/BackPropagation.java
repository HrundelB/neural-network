package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.methods.VecOptimization;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by afonin.s on 21.11.2014.
 */
public class BackPropagation<Loss extends TargetFunc & FuncC1> extends WeakListenerHolderImpl<Perceptron>
        implements VecOptimization<Loss> {

  private final double w;
  private final int numberOfSteps;
  private final Perceptron perceptron;

  public BackPropagation(int[] dims, Function activationFunction, double w, int numberOfSteps) {
    if (dims.length < 2) {
      throw new IllegalArgumentException("Perceptron must have at least two dims: input and output");
    }
    Mx[] mxes = new Mx[dims.length - 1];
    for (int l = 0; l < dims.length - 1; l++) {
      mxes[l] = new VecBasedMx(dims[l + 1], dims[l]);
      fillWithRandom(mxes[l]);
    }
    this.perceptron = new Perceptron(mxes, activationFunction);
    this.w = w;
    this.numberOfSteps = numberOfSteps;
  }

  @Override
  public Perceptron fit(VecDataSet learn, Loss loss) {
    double w = this.w;
    for (int k = 0; k < numberOfSteps; k++) {
      for (int d = 0; d < learn.length(); d++) {
        int index = new Random().nextInt(learn.length());
        Vec output = perceptron.trans(learn.at(index));

        Vec delta = loss.gradient(output);
        for (int l = perceptron.getCountOfLayers() - 1; l > 0; l--) {
          VecTools.scale(delta, perceptron.getActivationFunction().vecDerivative(perceptron.getSum(l)));
          VecTools.append(perceptron.getWeightMx(l), VecTools.scale(VecTools.outer(delta, perceptron.getOutput(l - 1)), w));
          delta = MxTools.multiply(MxTools.transpose(perceptron.getWeightMx(l)), delta);
        }
        VecTools.scale(delta, perceptron.getActivationFunction().vecDerivative(perceptron.getSum(0)));
        VecTools.append(perceptron.getWeightMx(0), VecTools.scale(VecTools.outer(delta, learn.at(index)), w));
      }
      invoke(perceptron);
      w = w / 1.07;
    }
    return perceptron;
  }

  public void save(String pathToFolder) throws IOException {
    for (int i = 0; i < perceptron.getCountOfLayers(); i++) {
      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, i));
      StringTools.printMx(perceptron.getWeightMx(i), file);
    }
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
