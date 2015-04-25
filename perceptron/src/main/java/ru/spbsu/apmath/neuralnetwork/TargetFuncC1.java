package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * Created by afonin.s on 23.04.2015.
 */
public abstract class TargetFuncC1 extends Func.Stub implements TargetFunc {

  protected final Vec target;
  protected final DataSet<?> owner;

  public TargetFuncC1(Vec target, DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  public abstract Vec gradient(Vec x, int indexOfLearningVec);

  public abstract double value(Vec x, int indexOfLearningVec);

  public double value(Mx mx) {
    double result = 0;
    for (int i = 0; i < mx.rows(); i++) {
      result += value(mx.row(i), i);
    }
    return result;
  }

  @Override
  public double value(Vec x) {
    return 0;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }
}
