package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import org.junit.BeforeClass;
import org.junit.Test;
import ru.spbsu.apmath.neuralnetwork.backpropagation.BackPropagation;
import ru.spbsu.apmath.neuralnetwork.backpropagation.FunctionC1;
import ru.spbsu.apmath.neuralnetwork.perceptron.LLLogit;
import ru.spbsu.apmath.neuralnetwork.perceptron.Perceptron;

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
  public static LLLogit logit;
  public static LLLogit testLogit;

  @BeforeClass
  public static void init() throws IOException {
    Pool<?> pool = DataTools.loadFromFeaturesTxt("src/test/data/features.txt.gz");
    dataSet = pool.vecData();
    logit = pool.target(LLLogit.class);

    Pool<?> testPool = DataTools.loadFromFeaturesTxt("src/test/data/featuresTest.txt.gz");
    testDataSet = testPool.vecData();
    testLogit = testPool.target(LLLogit.class);

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
    System.out.println(String.format("result: %s", testLogit.value(perceptron.transAll(testDataSet).col(0))));
  }

  @Test
  public void backPropagationTest() throws IOException {
    Perceptron perceptron = new Perceptron(new int[]{50, 20, 100, 20, 1},
            getActivateFunction());
    final BackPropagation<LLLogit, Vec> backPropagation = new BackPropagation(perceptron, 10000, 0.01, 0.0000003, 0.05);
    final Action<Learnable> action = new Action<Learnable>() {
      private long time = System.currentTimeMillis();
      private Learnable oldPerceptron;
      private int n = 0;

      @Override
      public void invoke(Learnable perceptron) {
        if (n % 100 == 0) {
          double l = logit.value(perceptron.transAll(dataSet));
          long now;
          if (n % 1000 == 0) {
            double t = testLogit.value(perceptron.transAll(testDataSet));
            List<Double> distances = new ArrayList<Double>(perceptron.depth());
            if (oldPerceptron != null) {
              for (int i = 0; i < perceptron.depth(); i++) {
                distances.add(distance(oldPerceptron.weights(i), perceptron.weights(i)));
              }
            }
            oldPerceptron = perceptron.clone();
            now = System.currentTimeMillis();
            System.out.println(String.format("Log likelihood on learn: %s; on test: %s; (time: %s ms)\ndistance: %s", l, t, now - time, distances));
            backPropagation.setStep(backPropagation.getStep() / 2);
          } else {
            now = System.currentTimeMillis();
            System.out.println(String.format("Log likelihood on learn: %s; (time: %s ms)", l, now - time));
          }
          time = now;
        }
        n++;
        System.out.print(String.format("%s\r", n));
      }
    };
    backPropagation.addListener(action);
    System.out.println("Learning...");
    Learnable<Vec> learnable = backPropagation.fit(dataSet, logit);
    learnable.save("perceptron/src/test/data/perceptron");
  }

  public static FunctionC1 getActivateFunction() {
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
