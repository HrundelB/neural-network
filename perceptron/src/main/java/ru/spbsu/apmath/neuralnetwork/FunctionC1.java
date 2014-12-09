package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 06.10.2014
 * Time: 18:07
 */
public abstract class FunctionC1 extends Function {

  public abstract double derivative(double x);

  public Vec vecDerivative(Vec x) {
    VecBuilder vecBuilder = new VecBuilder(x.dim());
    for (int i = 0; i < x.dim(); i++)
      vecBuilder.append(derivative(x.get(i)));
    return vecBuilder.build();
  }
}
