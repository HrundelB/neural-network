package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.multiply;
import static ru.spbsu.apmath.neuralnetwork.StringTools.printMx;

/**
 * Created by afonin.s on 21.11.2014.
 */
public class BackPropagation<Loss extends StatBasedLoss> extends WeakListenerHolderImpl<Trans>
        implements VecOptimization<Loss> {

  private final int countOfLayers;
  private final double w;
  private final int numberOfSteps;
  private final Perceptron perceptron;

  public BackPropagation(int[] dims, Function activationFunction, double w, int numberOfSteps) {
    if (dims.length < 2) {
      throw new IllegalArgumentException("Perceptron must have at least two dims: input and output");
    }
    this.countOfLayers = dims.length - 1;
    Mx[] mxes = new Mx[countOfLayers];
    for (int l = 0; l < countOfLayers; l++) {
      mxes[l] = new VecBasedMx(dims[l + 1], dims[l]);
      fillWithRandom(mxes[l]);
    }
    this.perceptron = new Perceptron(mxes, activationFunction);
    this.w = w;
    this.numberOfSteps = numberOfSteps;
  }

  @Override
  public Perceptron fit(VecDataSet learn, Loss loss) {
    for (int k = 0; k < numberOfSteps; k++) {
      Mx[] deltaWMxes = new Mx[countOfLayers];
      for (int i = 0; i < countOfLayers; i++) {
        deltaWMxes[i] = new VecBasedMx(perceptron.getWeightMx(i).rows(), perceptron.getWeightMx(i).columns());
        VecTools.fill(deltaWMxes[i], 0);
      }

      for (int d = 0; d < learn.length(); d++) {
        int index = new Random().nextInt(learn.length());
        perceptron.trans(learn.at(index));
        Vec[] deltas = new Vec[countOfLayers];
        deltas[countOfLayers - 1] = getDeltaForLastLayer(loss.target().get(index));
        for (int l = countOfLayers - 2; l >= 0; l--) {
          deltas[l] = getDeltaForLayer(l, deltas[l + 1]);
        }
        deltaWMxes = addToDeltaWMxes(deltaWMxes, w, learn.at(d), deltas);
      }

      for (int i = 0; i < countOfLayers; i++)
        VecTools.append(perceptron.getWeightMx(i), deltaWMxes[i]);

      invoke(perceptron);
    }
    return perceptron;
  }

  public void save(String pathToFolder) throws IOException {
    for (int i = 0; i < countOfLayers; i++) {
      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, i));
      printMx(perceptron.getWeightMx(i), file);
    }
  }

  private Mx[] addToDeltaWMxes(Mx[] deltaWMxes, double w, Vec learningVec, Vec[] deltas) {
    deltaWMxes[0] = VecTools.sum(VecTools.scale(VecTools.outer(deltas[0], learningVec), w), deltaWMxes[0]);
    for (int i = 1; i < countOfLayers; i++)
      deltaWMxes[i] = VecTools.sum(VecTools.scale(VecTools.outer(deltas[i], perceptron.getOutput(i - 1)), w), deltaWMxes[i]);
    return deltaWMxes;
  }

  private Vec getDeltaForLastLayer(double answer) {
    double result;
    double o = perceptron.getOutput(countOfLayers - 1).get(0);
    if (answer == 1) {
      result = 1 - o;
    } else {
      if (answer == -1) {
        result = -o;
      } else {
        throw new IllegalArgumentException("Answer must be only 1 or -1");
      }
    }
    return new ArrayVec(result);
  }

  private Vec getDeltaForLayer(int l, Vec previousDelta) {
    int neurons = perceptron.getWeightMx(l).rows();
    VecBuilder vecBuilder = new VecBuilder(neurons);
    for (int j = 0; j < neurons; j++) {
      double o = perceptron.getOutput(l).get(j);
      Mx weights = perceptron.getWeightMx(l + 1);
      Vec vec = new ArrayVec(weights.col(j).toArray());
      vecBuilder.append(o * (1 - o) * multiply(vec, previousDelta));
    }
    return vecBuilder.build();
  }

  private void fillWithRandom(Mx mx) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, Math.random());
      }
    }
  }
}
