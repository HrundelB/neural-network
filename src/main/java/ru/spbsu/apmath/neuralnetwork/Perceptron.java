package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.*;
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
      Vec[] rows = new Vec[dims[l + 1]];
      for (int i = 0; i < dims[l + 1]; i++) {
        VecBuilder vecBuilder = new VecBuilder(dims[l]);
        for (int j = 0; j < dims[l]; j++) {
          vecBuilder.append(Math.random());
        }
        rows[i] = vecBuilder.build();
      }
      Mx w = new RowsVecArrayMx(rows);
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
      File file = new File(String.format("%s%smatrix%s.txt", pathToFolder, System.lineSeparator(), i));
      printMx(layers[i].getWeights(), file);
    }
  }

  public double calculate(Vec input) {
    layers[0].calculate(input);
    for (int i = 1; i < countOfLayers; i++)
      layers[i].calculate(layers[i - 1].getOutputs());
    return layers[countOfLayers - 1].getOutputs().get(0);
  }

  public void backPropagation(double w, Vec[] learning, int[] answers, int numberOfSteps) {
    int countOfLearningSamples = learning.length;
    System.out.println("Learning...");
    for (int k = 0; k < numberOfSteps; k++) {

      logLikelihood = 0;
      Mx[] deltaWMxes = new Mx[countOfLayers];
      for (int i = 0; i < countOfLayers; i++) {
        int rows = layers[i].getWeights().rows();
        int columns = layers[i].getWeights().columns();
        Vec[] vecRows = new Vec[rows];
        for (int j = 0; j < rows; j++) {
          VecBuilder vecBuilder = new VecBuilder();
          for (int h = 0; h < columns; h++) {
            vecBuilder.append(0);
          }
          vecRows[j] = vecBuilder.build();
        }
        deltaWMxes[i] = new RowsVecArrayMx(vecRows);
      }

      for (int d = 0; d < countOfLearningSamples; d++) {
        int index = new Random().nextInt(countOfLearningSamples);
        this.calculate(learning[index]);
        Vec[] deltas = new Vec[countOfLayers];
        deltas[countOfLayers - 1] = getDeltaForLastLayer(answers[index]);
        for (int l = countOfLayers - 2; l >= 0; l--) {
          deltas[l] = getDeltaForLayer(l, deltas[l + 1]);
        }
        deltaWMxes = addToDeltaWMxes(deltaWMxes, w, learning[d], deltas);
      }
      System.out.println("Log likelihood function: " + logLikelihood);

      for (int i = 0; i < countOfLayers; i++)
        append(layers[i].getWeights(), deltaWMxes[i]);
    }
  }

  private Mx[] addToDeltaWMxes(Mx[] deltaWMxes, double w, Vec learningVec, Vec[] deltas) {
    deltaWMxes[0] = sum(scale(multiplyVecs(deltas[0], learningVec), w), deltaWMxes[0]);
    for (int i = 1; i < countOfLayers; i++)
      deltaWMxes[i] = sum(scale(multiplyVecs(deltas[i], layers[i - 1].getOutputs()), w), deltaWMxes[i]);
    return deltaWMxes;
  }

  private Vec getDeltaForLastLayer(int answer) {
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

  private Mx multiplyVecs(Vec a, Vec b) {
    Mx result = new VecBasedMx(a.dim(), b.dim());
    for (int i = 0; i < result.rows(); i++) {
      for (int j = 0; j < result.columns(); j++) {
        result.set(i, j, a.get(i) * b.get(j));
      }
    }
    return result;
  }
}
