package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.ml.data.set.DataSet;
import org.junit.BeforeClass;
import org.junit.Test;
import ru.spbsu.apmath.neuralnetwork.automaton.MultiLLLogit;
import ru.spbsu.apmath.neuralnetwork.automaton.ProbabilisticAutomaton;
import ru.spbsu.apmath.neuralnetwork.backpropagation.BackPropagation;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static ru.spbsu.apmath.neuralnetwork.PerceptronTest.getActivateFunction;
import static ru.spbsu.apmath.neuralnetwork.StringTools.loadTrainTxt;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomatonTest {

  private static MyPool<CharSeq> pool;

  @BeforeClass
  public static void init() throws IOException {
    pool = loadTrainTxt("perceptron/src/test/data/train.txt.gz");
  }

  @Test
  public void manualTest() {
    ProbabilisticAutomaton probabilisticAutomaton = new ProbabilisticAutomaton(3, 2, new Character[]{'a', 'b'}, getActivateFunction());
    final DataSet<CharSeq> dataSet = new DataSet.Stub<CharSeq>(null) {
      private CharSeq[] charSeqs = new CharSeq[]{
              new CharSeqArray(new char[]{'a', 'b', 'b'}),
              new CharSeqArray(new char[]{'b', 'a', 'a'}),
              new CharSeqArray(new char[]{'a', 'b'}),
              new CharSeqArray(new char[]{'a', 'b', 'b', 'a'})
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
    Vec target = new ArrayVec(3, 4, 3, 4);
    final MultiLLLogit multiLLLogit = new MultiLLLogit(5, target, dataSet);
    BackPropagation<MultiLLLogit, CharSeq> backPropagation = new BackPropagation<MultiLLLogit, CharSeq>(probabilisticAutomaton, 100000, 0.001, 0.00003, 0.2);
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

  public Character[] findCharacters() throws IOException {
    Set<Character> set = new HashSet<Character>();
    for (int i = 0; i < pool.getDataSet().length(); i++) {
      CharSeq charSeq = pool.getDataSet().at(i);
      for (int j = 0; j < charSeq.length(); j++) {
        set.add(charSeq.at(j));
      }
    }
    System.out.println(set);
    return set.toArray(new Character[]{});
  }

  @Test
  public void test() throws IOException {
    int states = 10;
    ProbabilisticAutomaton probabilisticAutomaton = new ProbabilisticAutomaton(states, 2, findCharacters(), getActivateFunction());
    for (int i = 0; i < pool.getTarget().length(); i++) {
      pool.getTarget().adjust(i, states);
    }
    final MultiLLLogit multiLLLogit = new MultiLLLogit(states + 2, pool.getTarget(), pool.getDataSet());
    BackPropagation<MultiLLLogit, CharSeq> backPropagation = new BackPropagation<MultiLLLogit, CharSeq>(probabilisticAutomaton, 3000, 0.001, 0.00003, 0.2);
    Action<Learnable> action = new Action<Learnable>() {
      private int n = 0;

      @Override
      public void invoke(Learnable learnable) {
        if (n % 100 == 0) {
          double l = multiLLLogit.value(learnable.transAll(pool.getDataSet()));
          System.out.println(String.format("Log likelihood on learn: %s", l));
          int index = new Random().nextInt(pool.getDataSet().length());
          System.out.println(String.format("index: %s, target: %s, compute: %s", index,
                  pool.getTarget().get(index), learnable.compute(pool.getDataSet().at(index))));
        }
        n++;
        System.out.print(String.format("%s\r", n));
      }
    };
    backPropagation.addListener(action);
    backPropagation.fit(pool.getDataSet(), multiLLLogit);
  }
}
