package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.converters.Double2BufferConverter;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.DataSet;

import java.util.ArrayList;
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

  public MyPool(List<T> data, List<Double> target) {
    this(data, new ArrayVec(target.size()));
    for (int i = 0; i < target.size(); i++) {
      this.target.set(i, target.get(i));
    }
  }

  public MyPool(final List<T> data, Vec target) {
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
    this.target = target;
  }

  public DataSet<T> getDataSet() {
    return dataSet;
  }

  public Vec getTarget() {
    return target;
  }

  public int size() {
    if (target.length() == dataSet.length()) {
      return target.length();
    } else {
      throw new RuntimeException(String.format("Lenght of dataSet and target must be equal! DataSet: %s, target: %s",
              dataSet.length(), target.length()));
    }
  }

  public static MyPool<CharSeq> getBalancedPool(Pair<List<CharSeq>, List<Double>> pair, int offset, int step) {
    List<CharSeq> data = new ArrayList<CharSeq>();
    List<Double> target = new ArrayList<Double>();
    int n0 = 0, n1 = 0;
    for (int i = offset; i < pair.getSecond().size(); i += step) {
      double answer = pair.getSecond().get(i);
      if (answer == 0 && n1 >= n0) {
        data.add(pair.getFirst().get(i));
        target.add(answer);
        n0++;
      } else if (answer > 0 && n0 >= n1) {
        data.add(pair.getFirst().get(i));
        target.add(answer);
        n1++;
      }
    }
    System.out.println(String.format("All: %s, zero: %s, other: %s", target.size(), n0, n1));
    return new MyPool<CharSeq>(data, target);
  }
}
