package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import org.junit.Test;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomatonTest {

  @Test
  public void test() {
    ProbabilisticAutomaton probabilisticAutomaton = new ProbabilisticAutomaton(3, 2, new Character[] {'a', 'b'});
    System.out.println(probabilisticAutomaton.getWeights('a'));
    System.out.println(probabilisticAutomaton.getWeights('b'));
    System.out.println(probabilisticAutomaton.calculate("ba"));
  }
}
