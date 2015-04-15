package ru.spbsu.apmath.neuralnetwork;

import org.junit.Test;
import ru.spbsu.apmath.neuralnetwork.automaton.ProbabilisticAutomaton;

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
