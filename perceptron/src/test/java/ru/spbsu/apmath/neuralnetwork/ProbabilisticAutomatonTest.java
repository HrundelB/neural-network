package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.DataSet;
import org.junit.BeforeClass;
import org.junit.Test;
import ru.spbsu.apmath.neuralnetwork.automaton.MultiLLLogit;
import ru.spbsu.apmath.neuralnetwork.automaton.ProbabilisticAutomaton;
import ru.spbsu.apmath.neuralnetwork.backpropagation.BackPropagation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

import static ru.spbsu.apmath.neuralnetwork.PerceptronTest.getActivateFunction;
import static ru.spbsu.apmath.neuralnetwork.StringTools.loadTrainTxt;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomatonTest {

  private static MyPool<CharSeq> pool;

  @BeforeClass
  public static void init() throws IOException {
    Pair<List<CharSeq>, List<Double>> pair = loadTrainTxt("perceptron/src/test/data/train.txt.gz");
    List<CharSeq> data = new ArrayList<CharSeq>();
    List<Double> target = new ArrayList<Double>();
    int n0 = 0, n1 = 0;
    for (int i = 0; i < pair.getSecond().size(); i += 50) {
      double answer = pair.getSecond().get(i);
      if (answer == 0 && n1 >= n0) {
        data.add(pair.getFirst().get(i));
        target.add(answer);
        n0++;
      } else if (answer > 0 && n0 >= n1) {
        data.add(pair.getFirst().get(i));
        target.add(answer);
        n1++;
      }
    }
    pool = new MyPool<CharSeq>(data, target);
    int n = 0;
    int length = pool.getTarget().length();
    for (int i = 0; i < length; i++) {
      if (pool.getTarget().get(i) > 0) {
        n++;
      }
    }
    System.out.println(String.format("All: %s, zero: %s, other: %s", length, length - n, n));
  }

  @Test
  public void manualTest() throws IOException {
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
    Vec target = new ArrayVec(3, 3, 4, 4);
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
    probabilisticAutomaton.save("perceptron/src/test/data/automaton");
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
    BackPropagation<MultiLLLogit, CharSeq> backPropagation = new BackPropagation<MultiLLLogit, CharSeq>(probabilisticAutomaton, 3000, 0.01, 0.003, 0.05);
    Action<Learnable> action = new Action<Learnable>() {
      private int n = 0;

      @Override
      public void invoke(Learnable learnable) {
        if (n % 100 == 0) {
          double l = multiLLLogit.value(learnable.transAll(pool.getDataSet()));
          System.out.println(String.format("Log likelihood on learn: %s", l));
          int index = new Random().nextInt(pool.getDataSet().length());
          Vec compute = getComputedVec(learnable, index);
          System.out.println(String.format("index: %s, target: %s, compute: %s", index,
                  pool.getTarget().get(index), compute));
          System.out.println(pool.getDataSet().at(index));
          double perplexity = getPerplexity(learnable, multiLLLogit);
          System.out.println("Perplexity: " + perplexity);
        }
        n++;
        System.out.print(String.format("%s\r", n));
      }
    };
    backPropagation.addListener(action);
    backPropagation.fit(pool.getDataSet(), multiLLLogit);
    probabilisticAutomaton.save("perceptron/src/test/data/automaton");
  }

  private double getPerplexity(Learnable learnable, MultiLLLogit multiLLLogit) {
    double entropy = 0;
    int len = pool.getDataSet().length();
    double d = 1 / (double)len;
    for (int i = 0; i < len; i++) {
      double value = multiLLLogit.value((Vec) learnable.compute(pool.getDataSet().at(i)), i);
      entropy += d * value;
    }
    return Math.exp(-entropy);
  }

  private Vec getComputedVec(Learnable learnable, int index) {
    Vec vec = (Vec) learnable.compute(pool.getDataSet().at(index));
    Vec compute = new ArrayVec(vec.length() + 1);
    double sum = 1;
    for (int j = 0; j < vec.length(); j++) {
      sum += vec.get(j);
    }
    for (int i = 0; i < vec.length(); i++) {
      compute.set(i, vec.get(i) / sum);
    }
    compute.set(vec.length(), 1 / sum);
    return compute;
  }

  @Test
  public void writeSamples() throws FileNotFoundException {
    File samples = new File("perceptron/src/test/data/sample.txt");
    File stest = new File("perceptron/src/test/data/stest.txt");
    PrintWriter samplesPrintWriter = new PrintWriter(samples);
    PrintWriter stestPrintWriter = new PrintWriter(stest);
    for (int i = 0; i < pool.getDataSet().length(); i++) {
      samplesPrintWriter.println(String.format("%s %s", pool.getDataSet().at(i), (int) pool.getTarget().get(i)));
      stestPrintWriter.println(pool.getDataSet().at(i));
    }
    samplesPrintWriter.close();
    stestPrintWriter.close();
  }

  @Test
  public void getAutomatonTest() throws IOException {
    final ProbabilisticAutomaton probabilisticAutomaton = ProbabilisticAutomaton.getAutomatonByFiles("perceptron/src/test/data/automaton", getActivateFunction());
    final MultiLLLogit multiLLLogit = new MultiLLLogit(12, pool.getTarget(), pool.getDataSet());
    double perplexity = getPerplexity(probabilisticAutomaton, multiLLLogit);
    System.out.println("Perplexity: " + perplexity);
  }
}
