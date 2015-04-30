package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;

import java.util.List;

/**
 * Created by afonin.s on 30.04.2015.
 */
public class MyPool<T extends Seq> {
  private DataSet<T> dataSet;
  private Vec target;

  public MyPool(DataSet<T> dataSet, Vec target) {
    this.dataSet = dataSet;
    this.target = target;
  }

  public MyPool(final List<T> data, List<Double> target) {
    this.dataSet = new DataSet.Stub<T>(null) {
      @Override
      public T at(int i) {
        return data.get(i);
      }

      @Override
      public int length() {
        return data.size();
      }

      @Override
      public Class<T> elementType() {
        return null;
      }
    };
    this.target = new ArrayVec(target.size());
    for (int i = 0; i < target.size(); i++) {
      this.target.set(i, target.get(i));
    }
  }

  public DataSet<T> getDataSet() {
    return dataSet;
  }

  public Vec getTarget() {
    return target;
  }
}
