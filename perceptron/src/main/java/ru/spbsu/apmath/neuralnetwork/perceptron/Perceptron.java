package ru.spbsu.apmath.neuralnetwork.perceptron;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import ru.spbsu.apmath.neuralnetwork.Learnable;
import ru.spbsu.apmath.neuralnetwork.StringTools;
import ru.spbsu.apmath.neuralnetwork.backpropagation.Function;
import ru.spbsu.apmath.neuralnetwork.backpropagation.FunctionC1;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import static com.spbsu.commons.math.vectors.MxTools.multiply;
import static ru.spbsu.apmath.neuralnetwork.StringTools.readMx;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 22:23
 */
public class Perceptron extends Learnable<Vec> {

  private final Mx[] weightMxes;
  private final Vec[] outputs;
  private final Vec[] sums;
  private final Function activationFunction;
  private final Random random = new FastRandom();

  public Perceptron(Mx[] mxes, Function activationFunction) {
    this.weightMxes = mxes;
    this.outputs = new Vec[mxes.length + 1];
    this.sums = new Vec[mxes.length];
    for (int i = 0; i < mxes.length; i++) {
      if (i < mxes.length - 1 && mxes[i].rows() != mxes[i + 1].columns()) {
        throw new IllegalArgumentException(
                String.format("Count of rows (%s) of %s matrix isn't equal to count of columns (%s) of %s matrix",
                        mxes[i].rows(), i, mxes[i + 1].columns(), i + 1));
      }
    }
    this.activationFunction = activationFunction;
  }

  public Perceptron(int[] dims, FunctionC1 activationFunction) {
    if (dims.length < 2) {
      throw new IllegalArgumentException("Perceptron must have at least two dims: input and output");
    }
    weightMxes = new Mx[dims.length - 1];
    for (int l = 0; l < dims.length - 1; l++) {
      weightMxes[l] = new VecBasedMx(dims[l + 1], dims[l]);
      fillWithRandom(weightMxes[l]);
    }
    this.activationFunction = activationFunction;
    this.outputs = new Vec[dims.length];
    this.sums = new Vec[weightMxes.length];
  }

  @Override
  public void setLearn(Vec learn) {
  }


  @Override
  public Vec compute(Vec argument) {
    outputs[0] = argument;
    for (int i = 0; i < weightMxes.length; i++) {
      sums[i] = multiply(weightMxes[i], outputs[i]);
      outputs[i + 1] = activationFunction.vecValue(sums[i]);
    }
    return outputs[outputs.length - 1];
  }

  @Override
  public Vec getOutput(int index) {
    return outputs[index + 1];
  }

  @Override
  public Vec getSum(int index) {
    return sums[index];
  }

  @Override
  public void save(String pathToFolder) throws IOException {
    for (int i = 0; i < weightMxes.length; i++) {
      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, i));
      StringTools.printMx(weightMxes[i], file);
    }
  }

  public static Perceptron getPerceptronByFiles(Function activationFunction, String... paths) throws IOException {
    Mx[] mxes = new Mx[paths.length];
    for (int i = 0; i < paths.length; i++) {
      mxes[i] = readMx(new File(paths[i]));
    }
    return new Perceptron(mxes, activationFunction);
  }

  @Override
  public int depth() {
    return weightMxes.length;
  }

  @Override
  public int ydim() {
    return weightMxes[weightMxes.length - 1].rows();
  }

  @Override
  public Mx weights(int i) {
    return weightMxes[i];
  }

  @Override
  public Perceptron clone() {
    Mx[] mxes = new Mx[weightMxes.length];
    for (int i = 0; i < mxes.length; i++) {
      mxes[i] = new VecBasedMx(weightMxes[i]);
    }
    return new Perceptron(mxes, activationFunction);
  }

  private void fillWithRandom(Mx mx) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, random.nextGaussian());
      }
    }
  }
}
