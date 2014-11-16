package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import org.hamcrest.Description;
import org.hamcrest.Factory;
import org.hamcrest.TypeSafeMatcher;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 29.10.2014
 * Time: 1:07
 */
public class MxEqualsMatcher extends TypeSafeMatcher<Mx> {

  private Mx originalMx;

  public MxEqualsMatcher(Mx originalMx) {
    this.originalMx = originalMx;
  }

  @Override
  protected boolean matchesSafely(Mx mx) {
    if (mx.rows() != originalMx.rows() || mx.columns() != originalMx.columns()) {
      return false;
    } else {
      for (int i = 0; i < mx.rows(); i++) {
        for (int j = 0; j < mx.columns(); j++) {
          if (mx.get(i,j) != originalMx.get(i, j)) {
            return false;
          }
        }
      }
      return true;
    }
  }

  @Override
  public void describeTo(Description description) {
    description.appendText("матрицы эквивалентны");
  }

  @Override
  protected void describeMismatchSafely(Mx item, Description mismatchDescription) {
    mismatchDescription.appendText("матрица\n").appendValue(originalMx).appendText("\nне эквивалентна матрице\n").
            appendValue(item);
  }

  @Factory
  public static MxEqualsMatcher mxEqualsTo(Mx matrix) {
    return new MxEqualsMatcher(matrix);
  }
}
