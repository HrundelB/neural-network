package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
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
  public static Mx dataSet;
  public static Vec answers;

  @BeforeClass
  public static void init() throws IOException {
    Pool<?> pool = DataTools.loadFromFeaturesTxt("jmll/ml/src/test/resources/com/spbsu/ml/features.txt.gz");
    dataSet = pool.vecData().data();
    answers = pool.target(L2.class).target();
    for (int i = 0; i < answers.dim(); i++) {
      if (answers.get(i) > 0.07) {
        answers.set(i, 1);
      } else {
        answers.set(i, -1);
      }
    }
    System.out.println(String.format("dataSet: rows - %s, columns - %s; answers: %s", dataSet.rows(), dataSet.columns(),
            answers.dim()));
  }

  @Test
  public void backPropagationTest() throws IOException {
    Perceptron perceptron = new Perceptron(new int[] {50,100,1});
    perceptron.backPropagation(0.00008, dataSet, answers, 20);
    perceptron.save("perceptron/src/test/data/perceptron");
  }

  @Test
  public void calculateTest() throws IOException {
    Perceptron perceptron = Perceptron.getPerceptronByFiles("perceptron/src/test/data/perceptron/matrix0.txt",
            "perceptron/src/test/data/perceptron/matrix1.txt");

    System.out.println("On learning data set:");
    for (int i = 0; i < 1000; i+= 100) {
      double result = perceptron.calculate(dataSet.row(i));
      System.out.println(String.format("[row %s] answer: %s, result: %s", i, answers.get(i), result));
    }

    System.out.println("On test data set:");
    Pool<?> pool = DataTools.loadFromFeaturesTxt("jmll/ml/src/test/resources/com/spbsu/ml/featuresTest.txt.gz");
    Mx testDataSet = pool.vecData().data();
    Vec testAnswers = pool.target(L2.class).target();
    for (int i = 0; i < 1000; i+= 100) {
      double result = perceptron.calculate(testDataSet.row(i));
      System.out.println(String.format("[row %s] answer: %s, result: %s", i, testAnswers.get(i), result));
    }
  }
}
