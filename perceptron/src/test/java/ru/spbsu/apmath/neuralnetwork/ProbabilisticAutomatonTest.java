package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.ml.data.set.DataSet;
import org.junit.Test;
import ru.spbsu.apmath.neuralnetwork.automaton.MultiLLLogit;
import ru.spbsu.apmath.neuralnetwork.automaton.ProbabilisticAutomaton;
import ru.spbsu.apmath.neuralnetwork.backpropagation.BackPropagation;

import static ru.spbsu.apmath.neuralnetwork.PerceptronTest.getActivateFunction;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomatonTest {

  @Test
  public void test() {
    ProbabilisticAutomaton probabilisticAutomaton = new ProbabilisticAutomaton(3, 2, new Character[] {'a', 'b'}, getActivateFunction());
    final DataSet<CharSeq> dataSet = new DataSet.Stub<CharSeq>(null) {
      private CharSeq[] charSeqs = new CharSeq[] {
              new CharSeqArray(new char[] {'a', 'b', 'b'}),
              new CharSeqArray(new char[] {'b', 'a', 'a'}),
              new CharSeqArray(new char[] {'a', 'b'}),
              new CharSeqArray(new char[] {'a', 'b', 'b', 'a'})
      };

      @Override
      public CharSeq at(int i) {
        return charSeqs[i];
      }

      @Override
      public int length() {
        return charSeqs.length;
      }

      @Override
      public Class<CharSeq> elementType() {
        return CharSeq.class;
      }
    };
    Vec target = new ArrayVec(4, 5, 4, 5);
    final MultiLLLogit multiLLLogit = new MultiLLLogit(6, target, dataSet);
    BackPropagation<MultiLLLogit, CharSeq> backPropagation = new BackPropagation<MultiLLLogit, CharSeq>(probabilisticAutomaton, 1000, 0.01, 0.003, 0.2);
    Action<Learnable> action = new Action<Learnable>() {
      @Override
      public void invoke(Learnable learnable) {
        Mx mx = learnable.transAll(dataSet);
        System.out.println(multiLLLogit.value(mx));
      }
    };
    backPropagation.addListener(action);
    backPropagation.fit(dataSet, multiLLLogit);
    System.out.println("============");
    System.out.println(probabilisticAutomaton.transAll(dataSet));
  }
}
