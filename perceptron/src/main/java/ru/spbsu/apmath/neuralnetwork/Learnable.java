package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;
import ru.spbsu.apmath.neuralnetwork.backpropagation.FunctionC1;

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

  public abstract int getComputedClass(L data);

  public abstract void save(String pathToFolder) throws IOException;

  public Mx transAll(DataSet<L> dataSet) {
    final Mx result = new VecBasedMx(ydim(), new ArrayVec(dataSet.length() * ydim()));
    for (int i = 0; i < dataSet.length(); i++) {
      VecTools.assign(result.row(i), compute(dataSet.at(i)));
    }
    return result;
  }
}
