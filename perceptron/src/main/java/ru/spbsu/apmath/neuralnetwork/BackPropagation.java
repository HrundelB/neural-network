package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.methods.VecOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.spbsu.commons.math.vectors.VecTools.*;

/**
 * Created by afonin.s on 21.11.2014.
 */
public class BackPropagation<Loss extends Logit> extends WeakListenerHolderImpl<Perceptron>
        implements VecOptimization<Loss> {

  private final int numberOfSteps;
  private final int[] dims;
  private final FunctionC1 activationFunction;
  private final double step;
  private final double alpha;
  private final double betta;

  private final Random random = new FastRandom();
  private final ExecutorService executorService;

  public BackPropagation(int[] dims, FunctionC1 activationFunction, int numberOfSteps, double step, double alpha,
                         double betta) {
    if (dims.length < 2) {
      throw new IllegalArgumentException("Perceptron must have at least two dims: input and output");
    }
    this.dims = dims;
    this.activationFunction = activationFunction;
    this.numberOfSteps = numberOfSteps;
    this.step = step;
    this.alpha = alpha;
    this.betta = betta;
    this.executorService = Executors.newFixedThreadPool(4);
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

  private Perceptron step(final VecDataSet learn, final Loss loss, final Perceptron perceptron) {
    List<Callable<Object>> tasks = new ArrayList<Callable<Object>>(1000);
    for (int t = 0; t < 1000; t++) {
      final int index = random.nextInt(learn.length());
      Callable<Object> callable = new Callable<Object>() {
        @Override
        public Object call() throws Exception {
          innerStep(learn, loss, perceptron, index);
          return null;
        }
      };
      tasks.add(callable);
    }
    try {
      executorService.invokeAll(tasks);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
    return perceptron;
  }

  private void innerStep(VecDataSet learn, Loss loss, Perceptron perceptron, int index) {
    Perceptron tmpPerceptron = perceptron.clone();
    for (int i = 0; i < tmpPerceptron.depth(); i++) {
      Mx mx = tmpPerceptron.weights(i);
      setZeroToMx(mx);
    }

    final Vec learningVec = learn.at(index);
    tmpPerceptron.trans(learningVec);
    final int depth = tmpPerceptron.depth() - 1;

    Vec delta;
    Mx[] mxes = new Mx[tmpPerceptron.depth()];

    delta = loss.gradient(tmpPerceptron.getSum(depth), index);
    mxes[depth] = scale(proj(outer(delta, tmpPerceptron.getOutput(depth - 1)), alpha), step);

    for (int l = depth - 1; l >= 0; l--) {
      delta = MxTools.multiply(MxTools.transpose(tmpPerceptron.weights(l + 1)), delta);
      scale(delta, function.vecValue(tmpPerceptron.getSum(l)));
      mxes[l] = scale(proj(outer(delta, tmpPerceptron.getOutput(l - 1)), alpha), step);
    }

    synchronized (perceptron) {
      for (int i = 0; i < mxes.length; i++) {
        append(perceptron.weights(i), mxes[i]);
      }
    }
  }

  private void setZeroToMx(Mx mx) {
    int l = (int) (1 / betta);
    int n = 0;
    while (n + 1 < mx.length()) {
      int index;
      if (n + l < mx.length()) {
        index = random.nextInt(l);
      } else {
        index = random.nextInt(mx.length() - n);
      }
      mx.set(n + index, 0);
      n += l;
    }
  }

  private void fillWithRandom(Mx mx) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, random.nextGaussian());
      }
    }
  }

  private Mx proj(Mx mx, double lambda) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, proj(mx.get(i, j), lambda));
      }
    }
    return mx;
  }

  private double proj(double x, double lambda) {
    if (Math.abs(x) < lambda) {
      return 0;
    } else if (x >= lambda) {
      return x - lambda;
    } else {
      return x + lambda;
    }
  }

  private Function function = new Function() {
    @Override
    public double call(double x) {
      return Math.exp(x) / Math.pow(1 + Math.exp(x), 2);
    }
  };
}
