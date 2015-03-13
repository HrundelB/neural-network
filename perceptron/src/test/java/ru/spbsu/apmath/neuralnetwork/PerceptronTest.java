package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Action;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static com.spbsu.commons.math.vectors.VecTools.distance;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 14.10.2014
 * Time: 23:48
 */
public class PerceptronTest {
  public static VecDataSet dataSet;
  public static VecDataSet testDataSet;
  public static Logit logit;
  public static Logit testLogit;

  @BeforeClass
  public static void init() throws IOException {
    Pool<?> pool = DataTools.loadFromFeaturesTxt("perceptron/src/test/data/features.txt.gz");
    dataSet = pool.vecData();
    logit = pool.target(Logit.class);

    Pool<?> testPool = DataTools.loadFromFeaturesTxt("perceptron/src/test/data/featuresTest.txt.gz");
    testDataSet = testPool.vecData();
    testLogit = testPool.target(Logit.class);

    System.out.println(String.format("dataSet: rows - %s, columns - %s", dataSet.data().rows(),
            dataSet.data().columns()));
    System.out.println(String.format("testDataSet: rows - %s, columns - %s", testDataSet.data().rows(),
            testDataSet.data().columns()));
  }

  @Test
  public void perceptronTransTest() throws IOException {
    Perceptron perceptron = Perceptron.getPerceptronByFiles(getActivateFunction(),
            "perceptron/src/test/data/perceptron/matrix0.txt",
            "perceptron/src/test/data/perceptron/matrix1.txt");
    System.out.println(String.format("result: %s", testLogit.value(perceptron.transAll(testDataSet.data()).col(0))));
  }

  @Test
  public void backPropagationTest() throws IOException {
    BackPropagation<Logit> backPropagation = new BackPropagation(new int[]{50, 100, 50, 100, 50, 1},
            getActivateFunction(), 100, 0.001, 0.0003, 0.8);
    final Action<Perceptron> action = new Action<Perceptron>() {
      private long time = System.currentTimeMillis();
      private Perceptron oldPerceptron;
      private int n = 0;

      @Override
      public void invoke(Perceptron perceptron) {
        double l = logit.value(perceptron.transAll(dataSet.data()).col(0));

        long now;
        if (n % 10 == 0) {
          double t = testLogit.value(perceptron.transAll(testDataSet.data()).col(0));
          List<Double> distances = new ArrayList<Double>(perceptron.depth());
          if (oldPerceptron != null) {
            for (int i = 0; i < perceptron.depth(); i++) {
              distances.add(distance(oldPerceptron.weights(i), perceptron.weights(i)));
            }
          }
          oldPerceptron = perceptron.clone();
          now = System.currentTimeMillis();
          System.out.println(String.format("Log likelihood on learn: %s; on test: %s; distance: %s (time: %s ms)", l, t, distances, now - time));
        } else {
          now = System.currentTimeMillis();
          System.out.println(String.format("Log likelihood on learn: %s; (time: %s ms)", l, now - time));
        }
        time = now;
        n++;
      }
    };
    backPropagation.addListener(action);
    System.out.println("Learning...");
    Perceptron perceptron = backPropagation.fit(dataSet, logit);
    perceptron.save("perceptron/src/test/data/perceptron");
  }

  private FunctionC1 getActivateFunction() {
    return new FunctionC1() {
      @Override
      public double derivative(double x) {
        return call(x) - call(x) * call(x);
      }

      @Override
      public double call(double x) {
        return 1. / (1. + Math.exp(-x));
      }
    };
  }
}
