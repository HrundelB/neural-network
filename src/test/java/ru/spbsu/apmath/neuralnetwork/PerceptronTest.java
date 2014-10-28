package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import org.junit.Test;

import java.io.IOException;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 14.10.2014
 * Time: 23:48
 */
public class PerceptronTest {

  @Test
  public void backPropagationTest() throws IOException {
    Perceptron perceptron = new Perceptron(new int[] {2,3,1});
    Vec[] learningVecs = new Vec[10];
    learningVecs[0] = new ArrayVec(1, 1);
    learningVecs[1] = new ArrayVec(0.5, 0.5);
    learningVecs[2] = new ArrayVec(0.5, 1);
    learningVecs[3] = new ArrayVec(0, 1);
    learningVecs[4] = new ArrayVec(0, 2);
    learningVecs[5] = new ArrayVec(5, 4);
    learningVecs[6] = new ArrayVec(4, 4);
    learningVecs[7] = new ArrayVec(5, 3);
    learningVecs[8] = new ArrayVec(4, 3);
    learningVecs[9] = new ArrayVec(3, 3);
    int[] answers = new int[10];
    for (int i = 0; i < 10; i++) {
      if (i < 5) {
        answers[i] = 1;
      } else {
        answers[i] = -1;
      }
    }
    perceptron.backPropagation(0.09, learningVecs, answers, 200);
    System.out.println("result for -1: " + perceptron.calculate(new ArrayVec(5, 4)));
    System.out.println("result for +1: " + perceptron.calculate(new ArrayVec(1, 1)));
    System.out.println("result for -1 (not learned): " + perceptron.calculate(new ArrayVec(4.5, 3.5)));
    System.out.println("result for +1 (not learned): " + perceptron.calculate(new ArrayVec(0.25, 1.25)));
    System.out.println("result for -1 (not learned): " + perceptron.calculate(new ArrayVec(5, 4.5)));
    System.out.println("result for +1 (not learned): " + perceptron.calculate(new ArrayVec(0, 0)));
    System.out.println("result for middle (not learned): " + perceptron.calculate(new ArrayVec(2, 2)));
    perceptron.save("src/test/data/perceptron");
  }
}
