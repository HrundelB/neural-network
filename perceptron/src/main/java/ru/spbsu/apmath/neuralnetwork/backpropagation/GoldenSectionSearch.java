package ru.spbsu.apmath.neuralnetwork.backpropagation;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 09.12.2014
 * Time: 20:16
 */
public class GoldenSectionSearch {
  public static final double PHI = (1 + Math.sqrt(5)) / 2;

  public static double searchMax(double a, double b, Function function, double eps) {
    if (b < a)
      throw new IllegalArgumentException("b must be greater than a!");
    double x1, x2 = 0, y1, y2 = 0;
    boolean flag = false;
    x1 = b - (b - a) / PHI;
    y1 = function.call(x1);
    do {
      if (flag) {
        x1 = b - (b - a) / PHI;
        y1 = function.call(x1);
      } else {
        x2 = a + (b - a) / PHI;
        y2 = function.call(x2);
      }
      if (y1 <= y2) {
        a = x1;
        x1 = x2;
        y1 = y2;
        flag = false;
      } else {
        b = x2;
        x2 = x1;
        y2 = y1;
        flag = true;
      }
    } while (b - a > eps);
    return (a + b) / 2;
  }
}
