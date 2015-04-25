package ru.spbsu.apmath.neuralnetwork.automaton;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.data.set.DataSet;
import ru.spbsu.apmath.neuralnetwork.TargetFuncC1;

/**
 * Created by afonin.s on 25.04.2015.
 */
public class MultiLLLogit extends TargetFuncC1 {

  private int dim;

  public MultiLLLogit(int dim, Vec target, DataSet<?> owner) {
    super(target, owner);
    this.dim = dim;
  }

  @Override
  public Vec gradient(Vec x, int indexOfLearningVec) {
    Vec result = new ArrayVec(dim());

    double expMS = 1;
    for (int j = 0; j < dim() - 1; j++) {
      expMS += Math.exp(x.get(j));
    }

    int answer = (int) target.get(indexOfLearningVec);
    if (answer == dim() - 1) {
      for (int i = 0; i < dim() - 1; i++) {
        result.set(i, -Math.exp(x.get(i)) / expMS);
      }
    } else {
      for (int i = 0; i < dim() - 1; i++) {
        if (i == answer) {
          result.set(i, (expMS - Math.exp(x.get(answer))) / expMS);
        } else {
          result.set(i, Math.exp(x.get(i)) / expMS);
        }
      }
    }
    result.set(dim() - 1, 0);
    return result;
  }

  @Override
  public double value(Vec x, int indexOfLearningVec) {
    int answer = (int) target.get(indexOfLearningVec);
    double result;
    if (answer == dim() - 1) {
      result = 1;
    } else {
      result = Math.exp(x.get(answer));
    }
    double expMS = 1;
    for (int i = 0; i < dim() - 1; i++) {
      expMS += Math.exp(x.get(i));
    }
    result = result / expMS;
    return Math.log(result);
  }

  @Override
  public int dim() {
    return dim;
  }
}
