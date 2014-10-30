package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 30.10.2014
 * Time: 21:14
 */
public class MatrixTools {
  public static Mx multiplyVecs(Vec a, Vec b) {
    Mx result = new VecBasedMx(a.dim(), b.dim());
    for (int i = 0; i < a.dim(); i++) {
      for (int j = 0; j < b.dim(); j++) {
        result.set(i, j, a.get(i) * b.get(j));
      }
    }
    return result;
  }

  public static void fill(Vec vec, MakeDouble makeDouble) {
    for (int i = 0; i < vec.dim(); i++)
      vec.set(i, makeDouble.make());
  }

  public static <T extends Vec> T multiplyWithDouble(T a, double b) {
    for (int i = 0; i < a.dim(); i++)
      a.set(i, a.get(i) * b);
    return a;
  }

  public static <T extends Vec> T sum(T a, T b) {
    if (a.dim() != b.dim())
      throw new IllegalArgumentException(String.format("a.dim (%s) isn't equal to b.dim (%s)", a.dim(), b.dim()));
    for (int i = 0; i < a.dim(); i++)
      a.adjust(i, b.get(i));
    return a;
  }

  public static MakeDouble makeRandomDouble = new MakeDouble() {
    @Override
    public double make() {
      return Math.random();
    }
  };

  public static MakeDouble makeZeroDouble = new MakeDouble() {
    @Override
    public double make() {
      return 0;
    }
  };
}
