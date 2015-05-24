package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.DataSet;
import ru.spbsu.apmath.neuralnetwork.backpropagation.FunctionC1;
import ru.spbsu.apmath.neuralnetwork.perceptron.LLLogit;

import java.io.IOException;

/**
 * Created by afonin.s on 15.04.2015.
 */
public abstract class Learnable<L extends Seq> implements Cloneable, Computable<L, Vec> {

  protected Vec[] outputs;
  protected Vec[] sums;
  protected final FunctionC1 activationFunction;

  public Learnable(FunctionC1 activationFunction) {
    this.activationFunction = activationFunction;
  }

  public Vec getSum(int i) {
    return sums[i];
  }

  public Vec getOutput(int i) {
    return outputs[i + 1];
  }

  public FunctionC1 getActivationFunction() {
    return activationFunction;
  }

  public abstract void setLearn(L learn);

  public abstract int depth();

  public abstract int ydim();

  public abstract Mx weights(int i);

  public abstract Learnable clone();

  public abstract void save(String pathToFolder) throws IOException;

  public Mx transAll(DataSet<L> dataSet) {
    final Mx result = new VecBasedMx(ydim(), new ArrayVec(dataSet.length() * ydim()));
    for (int i = 0; i < dataSet.length(); i++) {
      VecTools.assign(result.row(i), compute(dataSet.at(i)));
    }
    return result;
  }

  public double getAccuracy(DataSet<L> dataSet, LLLogit logit){
    int len = dataSet.length();
    int n = 0;
    for (int i = 0; i < len; i++) {
      if(compute(dataSet.at(i)).get(0) > 0.5) {
        if (logit.isPositive(i)) {
          n++;
        }
      } else {
        if (!logit.isPositive(i)) {
          n++;
        }
      }
    }
    return (double) n / (double) len;
  }

  public Pair<Double, Double> getPrecisionAndRecall(DataSet<L> dataSet, LLLogit logit, int category) {
    double tp = 0, tn = 0, fp = 0, fn = 0;
    for (int i = 0; i < dataSet.length(); i++) {
      int answer = (compute(dataSet.at(i)).get(0) > 0.5 ? 1 : 0);
      if (answer == category) {
        if (logit.target.get(i) == category) {
          tp++;
        } else {
          fp++;
        }
      } else {
        if (logit.target.get(i) == category) {
          fn++;
        } else {
          tn++;
        }
      }
    }
    double precision = tp /(tp + fp);
    double recall = tp / (tp + fn);
    return new Pair<>(precision, recall);
  }
}
