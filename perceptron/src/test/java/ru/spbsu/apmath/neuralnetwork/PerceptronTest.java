package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.loss.L2;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.util.Random;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 14.10.2014
 * Time: 23:48
 */
public class PerceptronTest {
  public static VecDataSet dataSet;
  public static Logit logit;
  public static Vec answers;

  @BeforeClass
  public static void init() throws IOException {
    Pool<?> pool = DataTools.loadFromFeaturesTxt("perceptron/src/test/data/features.txt.gz");
    dataSet = pool.vecData();
    logit = pool.target(Logit.class);
    answers = pool.target(L2.class).target();
    System.out.println(String.format("dataSet: rows - %s, columns - %s", dataSet.data().rows(),
            dataSet.data().columns()));
  }

  @Test
  public void perceptronTransTest() throws IOException {
    Perceptron perceptron = Perceptron.getPerceptronByFiles(getActivateFunction(),
            "perceptron/src/test/data/perceptron/matrix0.txt",
            "perceptron/src/test/data/perceptron/matrix1.txt");
    for (int i = 0; i < 10; i++) {
      int index = new Random().nextInt(dataSet.length());
      double result = perceptron.trans(dataSet.at(index)).get(0);
      System.out.println(String.format("Answer: %s, result: %s", answers.get(index), result));
    }
  }

  @Test
  public void backPropagationTest() throws IOException {
    BackPropagation<Logit> backPropagation = new BackPropagation(new int[]{50, 100, 1},
            getActivateFunction(), 10);
    final Action<Perceptron> action = new Action<Perceptron>() {
      private long time = System.currentTimeMillis();

      @Override
      public void invoke(Perceptron perceptron) {
        long now = System.currentTimeMillis();
        System.out.println(String.format("Log likelihood: %s (time: %s ms)",
                logit.value(perceptron.transAll(dataSet.data()).col(0)), now - time));
        time = now;
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
