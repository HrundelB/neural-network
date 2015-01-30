package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

import static com.spbsu.commons.math.vectors.VecTools.multiply;
import static java.lang.Math.log;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 23.11.2014
 * Time: 23:55
 */
public class Logit extends FuncC1.Stub implements TargetFunc {

  private final Vec target;
  private final DataSet<?> owner;

  public Logit(Vec target, DataSet<?> owner) {
    this.target = target;
    this.owner = owner;
  }

  @Override
  public Vec gradient(Vec x) {
    Vec result = new ArrayVec(x.dim());
    for (int i = 0; i < x.dim(); i++) {
      double expMS = Math.exp(x.get(i));
      if (isPositive(i)) // positive example
        result.set(i, 1 / (1 + expMS));
      else // negative
        result.set(i, -expMS / (1 + expMS));
    }
    return result;
  }

  public boolean isPositive(int i) {
    return target.get(i) > 0.07;
  }

  @Override
  public DataSet<?> owner() {
    return owner;
  }

  @Override
  public double value(Vec x) {
    double result = 0;
    for (int i = 0; i < x.dim(); i++) {
      if (isPositive(i)) // positive example
        result += log(x.get(i));
      else // negative
        result += log(1 - x.get(i));
    }
    return result;
  }

  @Override
  public int dim() {
    return target.dim();
  }
}
