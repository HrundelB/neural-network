package ru.spbsu.apmath.neuralnetwork;

import org.junit.Test;
import ru.spbsu.apmath.neuralnetwork.backpropagation.Function;
import ru.spbsu.apmath.neuralnetwork.backpropagation.GoldenSectionSearch;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 09.12.2014
 * Time: 22:20
 */
public class GoldenSectionSearchTest {

  @Test
  public void searchMaxTest() {
    Function function = new Function() {
      @Override
      public double call(double x) {
        return -x * x + 5 * x - 4;
      }
    };
    double max = GoldenSectionSearch.searchMax(0, 3, function, 0.00001);
    System.out.println(max);
  }

  @Test
  public void searchMaxSinTest() {
    Function function = new Function() {
      @Override
      public double call(double x) {
        return Math.sin(x) + 1/2 * Math.cos(5 * x - 3);
      }
    };
    double max = GoldenSectionSearch.searchMax(-3, 0, function, 0.00001);
    System.out.println(max);
  }
}
