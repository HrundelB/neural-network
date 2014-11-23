package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.ml.FuncC1;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 18:07
 */
public abstract class Function extends FuncC1.Stub {
  public abstract double call(double x);

  public abstract double derivative(double x);

  public Vec vecValue(Vec x) {
    VecBuilder vecBuilder = new VecBuilder(x.dim());
    for (int i = 0; i < x.dim(); i++)
      vecBuilder.append(call(x.get(i)));
    return vecBuilder.build();
  }

  public Vec vecDerivative(Vec x) {
    VecBuilder vecBuilder = new VecBuilder(x.dim());
    for (int i = 0; i < x.dim(); i++)
      vecBuilder.append(derivative(x.get(i)));
    return vecBuilder.build();
  }

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
