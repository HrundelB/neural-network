package ru.spbsu.apmath.neuralnetwork.backpropagation;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 09.12.2014
 * Time: 20:13
 */
public abstract class Function {

  public abstract double call(double x);

  public Vec vecValue(Vec x) {
    VecBuilder vecBuilder = new VecBuilder(x.dim());
    for (int i = 0; i < x.dim(); i++)
      vecBuilder.append(call(x.get(i)));
    return vecBuilder.build();
  }
}
