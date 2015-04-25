package ru.spbsu.apmath.neuralnetwork.perceptron;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.DataSet;
import ru.spbsu.apmath.neuralnetwork.TargetFuncC1;

import static java.lang.Math.log;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 23.11.2014
 * Time: 23:55
 */
public class LLLogit extends TargetFuncC1 {

  public LLLogit(Vec target, DataSet<?> owner) {
    super(target, owner);
  }

  @Override
  public Vec gradient(Vec x, int indexOfLearningVec) {
    Vec result = new ArrayVec(x.dim());
    for (int i = 0; i < x.dim(); i++) {
      if (isPositive(indexOfLearningVec)) {// positive example
        result.set(i, 1 / x.get(i));
      } else { // negative
        result.set(i, -1 / (1 - x.get(i)));
      }
    }
    return result;
  }

  public boolean isPositive(int i) {
    return target.get(i) > 0.07;
  }

  @Override
  public double value(Vec x, int indexOfLearningVec) {
    if (isPositive(indexOfLearningVec)) // positive example
      return log(x.get(0));
    else // negative
      return log(1 - x.get(0));
  }

  @Override
  public int dim() {
    return 1;
  }
}
