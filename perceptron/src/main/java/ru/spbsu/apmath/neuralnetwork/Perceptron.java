package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.multiply;
import static ru.spbsu.apmath.neuralnetwork.StringTools.printMx;
import static ru.spbsu.apmath.neuralnetwork.StringTools.readMx;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 22:23
 */
public class Perceptron {
  private int countOfLayers;
  private Layer[] layers;
  private double logLikelihood;

  public Perceptron(Mx[] mxes) {
    this.countOfLayers = mxes.length;
    this.layers = new Layer[countOfLayers];
    for (int i = 0; i < countOfLayers; i++) {
      if (i < countOfLayers - 1 && mxes[i].rows() != mxes[i + 1].columns()) {
        throw new IllegalArgumentException(
                String.format("Count of rows (%s) of %s matrix isn't equal to count of columns (%s) of %s matrix",
                        mxes[i].rows(), i, mxes[i + 1].columns(), i + 1));
      }
      this.layers[i] = new Layer(mxes[i], getActivateFunction());
    }
  }

  public Perceptron(int[] dims) {
    if (dims.length < 2) {
      throw new IllegalArgumentException("Perceptron must have at least two dims: input and output");
    }
    this.countOfLayers = dims.length - 1;
    this.layers = new Layer[countOfLayers];
    for (int l = 0; l < countOfLayers; l++) {
      Mx w = new VecBasedMx(dims[l + 1], dims[l]);
      fillWithRandom(w);
      layers[l] = new Layer(w, getActivateFunction());
    }
  }

  public static Perceptron getPerceptronByFiles(String... paths) throws IOException {
    Mx[] mxes = new Mx[paths.length];
    for (int i = 0; i < paths.length; i++) {
      mxes[i] = readMx(new File(paths[i]));
    }
    return new Perceptron(mxes);
  }

  public void save(String pathToFolder) throws IOException {
    for (int i = 0; i < countOfLayers; i++) {
      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, i));
      printMx(layers[i].getWeights(), file);
    }
  }

  public double calculate(Vec input) {
    layers[0].calculate(input);
    for (int i = 1; i < countOfLayers; i++)
      layers[i].calculate(layers[i - 1].getOutputs());
    return layers[countOfLayers - 1].getOutputs().get(0);
  }

  public void backPropagation(double w, Mx learning, Vec answers, int numberOfSteps) {
    int countOfLearningSamples = learning.rows();
    System.out.println("Learning...");
    for (int k = 0; k < numberOfSteps; k++) {

      logLikelihood = 0;
      Mx[] deltaWMxes = new Mx[countOfLayers];
      for (int i = 0; i < countOfLayers; i++) {
        deltaWMxes[i] = new VecBasedMx(layers[i].getWeights().rows(), layers[i].getWeights().columns());
        VecTools.fill(deltaWMxes[i], 0);
      }

      for (int d = 0; d < countOfLearningSamples; d++) {
        int index = new Random().nextInt(countOfLearningSamples);
        this.calculate(learning.row(index));
        Vec[] deltas = new Vec[countOfLayers];
        deltas[countOfLayers - 1] = getDeltaForLastLayer(answers.get(index));
        for (int l = countOfLayers - 2; l >= 0; l--) {
          deltas[l] = getDeltaForLayer(l, deltas[l + 1]);
        }
        deltaWMxes = addToDeltaWMxes(deltaWMxes, w, learning.row(d), deltas);
      }
      System.out.println("Log likelihood function: " + logLikelihood);

      for (int i = 0; i < countOfLayers; i++)
        VecTools.append(layers[i].getWeights(), deltaWMxes[i]);

      w = w * 0.8;
    }
  }

  private Mx[] addToDeltaWMxes(Mx[] deltaWMxes, double w, Vec learningVec, Vec[] deltas) {
    deltaWMxes[0] = VecTools.sum(VecTools.scale(VecTools.outer(deltas[0], learningVec), w), deltaWMxes[0]);
    for (int i = 1; i < countOfLayers; i++)
      deltaWMxes[i] = VecTools.sum(VecTools.scale(VecTools.outer(deltas[i], layers[i - 1].getOutputs()), w), deltaWMxes[i]);
    return deltaWMxes;
  }

  private Vec getDeltaForLastLayer(double answer) {
    double result;
    double o = layers[countOfLayers - 1].getOutputs().get(0);
    if (answer == 1) {
      logLikelihood += Math.log(o);
      result = 1 - o;
    } else {
      if (answer == -1) {
        logLikelihood += Math.log(1 - o);
        result = -1 * o;
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

  private Function getActivateFunction() {
    return new Function() {
      @Override
      public double call(double x) {
        return 1 / (1 + Math.exp(-1 * x));
      }
    };
  }

  private void fillWithRandom(Mx mx) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, Math.random());
      }
    }
  }
}
