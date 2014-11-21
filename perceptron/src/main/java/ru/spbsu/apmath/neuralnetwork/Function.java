package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 18:07
 */
public abstract class Function extends FuncC1.Stub {
  public abstract double call(double x);

  public abstract double derivative(double x);

  @Override
  public Vec gradient(Vec x) {
    return new ArrayVec(derivative(x.get(0)));
  }

  @Override
  public double value(Vec x) {
    return call(x.get(0));
  }

  @Override
  public int dim() {
    return 1;
  }
}
