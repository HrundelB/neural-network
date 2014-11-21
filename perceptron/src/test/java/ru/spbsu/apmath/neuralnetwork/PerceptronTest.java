package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.loss.L2;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 14.10.2014
 * Time: 23:48
 */
public class PerceptronTest {
  public static VecDataSet dataSet;
  public static L2 l2;

  @BeforeClass
  public static void init() throws IOException {
    Pool<?> pool = DataTools.loadFromFeaturesTxt("jmll/ml/src/test/resources/com/spbsu/ml/features.txt.gz");
    dataSet = pool.vecData();
    l2 = pool.target(L2.class);
    for (int i = 0; i < l2.target().dim(); i++) {
      if (l2.target().get(i) > 0.07) {
        l2.target().set(i, 1);
      } else {
        l2.target().set(i, -1);
      }
    }
    System.out.println(String.format("dataSet: rows - %s, columns - %s; answers: %s", dataSet.data().rows(),
            dataSet.data().columns(), l2.target().dim()));
  }

  @Test
  public void backPropagationTest() throws IOException {
    BackPropagation backPropagation = new BackPropagation(new int[]{50, 100, 1}, getActivateFunction(), 0.00008, 20);
    Perceptron perceptron = backPropagation.fit(dataSet, l2);
    backPropagation.save("perceptron/src/test/data/perceptron");
  }


  @Test
  public void calculateTest() throws IOException {
    Perceptron perceptron = Perceptron.getPerceptronByFiles(getActivateFunction(),
            "perceptron/src/test/data/perceptron/matrix0.txt",
            "perceptron/src/test/data/perceptron/matrix1.txt");

    System.out.println("On learning data set:");
//    for (int i = 0; i < 1000; i += 100) {
//      double result = perceptron.calculate(dataSet.row(i));
//      System.out.println(String.format("[row %s] answer: %s, result: %s", i, answers.get(i), result));
//    }

    System.out.println("On test data set:");
    Pool<?> pool = DataTools.loadFromFeaturesTxt("jmll/ml/src/test/resources/com/spbsu/ml/featuresTest.txt.gz");
    Mx testDataSet = pool.vecData().data();
    Vec testAnswers = pool.target(L2.class).target();
//    for (int i = 0; i < 1000; i += 100) {
//      double result = perceptron.calculate(testDataSet.row(i));
//      System.out.println(String.format("[row %s] answer: %s, result: %s", i, testAnswers.get(i), result));
//    }
  }

  private Function getActivateFunction() {
    return new Function() {
      @Override
      public double derivative(double x) {
        return call(x) - call(x) * call(x);
      }

      @Override
      public double call(double x) {
        return 1 / (1 + Math.exp(-1 * x));
      }
    };
  }
}
