package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.util.Pair;
import org.junit.BeforeClass;
import org.junit.Test;
import ru.spbsu.apmath.neuralnetwork.automaton.MultiLLLogit;
import ru.spbsu.apmath.neuralnetwork.automaton.ProbabilisticAutomaton;
import ru.spbsu.apmath.neuralnetwork.backpropagation.BackPropagation;
import ru.spbsu.apmath.neuralnetwork.backpropagation.FunctionC1;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static ru.spbsu.apmath.neuralnetwork.MyPool.getBalancedPool;
import static ru.spbsu.apmath.neuralnetwork.StringTools.findCharacters;
import static ru.spbsu.apmath.neuralnetwork.StringTools.loadTrainTxt;
import static ru.spbsu.apmath.neuralnetwork.StringTools.printVec;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomatonTest {

  private static MyPool<CharSeq> pool;
  private static MyPool<CharSeq> manualPool;

  @BeforeClass
  public static void init() throws IOException {
    Pair<List<CharSeq>, List<Double>> pair = loadTrainTxt("perceptron/src/test/data/train.txt.gz");
    pool = getBalancedPool(pair);

    List<CharSeq> data = Arrays.asList(CharSeq.copy("abb"), CharSeq.copy("baa"), CharSeq.copy("ab"), CharSeq.copy("abba"));
    manualPool = new MyPool<>(data, new ArrayVec(3, 3, 4, 4));
  }

  @Test
  public void manualTest() throws IOException {
    ProbabilisticAutomaton probabilisticAutomaton = new ProbabilisticAutomaton(3, 2, new Character[]{'a', 'b'}, getActivateFunction());
    final MultiLLLogit multiLLLogit = new MultiLLLogit(5, manualPool.getTarget(), manualPool.getDataSet());
    BackPropagation<MultiLLLogit, CharSeq> backPropagation = new BackPropagation<>(probabilisticAutomaton, 100000, 0.001, 0.00003, 0.2);
    Action<Learnable> action = new Action<Learnable>() {
      @Override
      public void invoke(Learnable learnable) {
        Mx mx = learnable.transAll(manualPool.getDataSet());
        System.out.println(multiLLLogit.value(mx));
      }
    };
    backPropagation.addListener(action);
    backPropagation.fit(manualPool.getDataSet(), multiLLLogit);
    System.out.println("============");
    System.out.println(probabilisticAutomaton.transAll(manualPool.getDataSet()));
    probabilisticAutomaton.save("perceptron/src/test/data/automaton");
  }

  @Test
  public void test() throws IOException {
    int states = 10;
    ProbabilisticAutomaton probabilisticAutomaton = new ProbabilisticAutomaton(states, 2, findCharacters(pool.getDataSet()), getActivateFunction());
    for (int i = 0; i < pool.size(); i++) {
      pool.getTarget().adjust(i, states);
    }
    final MultiLLLogit multiLLLogit = new MultiLLLogit(states + 2, pool.getTarget(), pool.getDataSet());
    final BackPropagation<MultiLLLogit, CharSeq> backPropagation = new BackPropagation<>(probabilisticAutomaton, 3000, 0.01, 0.0003, 0.2);
    Action<Learnable> action = new Action<Learnable>() {
      private int n = 0;

      @Override
      public void invoke(Learnable learnable) {
        n++;
        System.out.print(String.format("%s\r", n));
        if (n % 100 == 0) {
          double l = multiLLLogit.value(learnable.transAll(pool.getDataSet()));
          System.out.println(String.format("Log likelihood on learn: %s", l));
          int index = new Random().nextInt(pool.size());
          Vec compute = ((ProbabilisticAutomaton) learnable).getComputedVec(pool.getDataSet().at(index));
          System.out.println(String.format("index: %s, target: %s, compute: %s", index,
                  pool.getTarget().get(index), compute));
          System.out.println(String.format("%s -> %s", pool.getDataSet().at(index), printVec(((ProbabilisticAutomaton) learnable).compute(pool.getDataSet().at(index)))));
          double perplexity = learnable.getPerplexity(pool.getDataSet(), multiLLLogit);
          System.out.println("Perplexity: " + perplexity);
        }
      }
    };
    backPropagation.addListener(action);
    backPropagation.fit(pool.getDataSet(), multiLLLogit);
    probabilisticAutomaton.save("perceptron/src/test/data/automaton");
  }

  @Test
  public void writeSamples() throws FileNotFoundException {
    File samples = new File("perceptron/src/test/data/sample.txt");
    File stest = new File("perceptron/src/test/data/stest.txt");
    PrintWriter samplesPrintWriter = new PrintWriter(samples);
    PrintWriter stestPrintWriter = new PrintWriter(stest);
    for (int i = 0; i < pool.size(); i++) {
      samplesPrintWriter.println(String.format("%s %s", pool.getDataSet().at(i), (int) pool.getTarget().get(i)));
      stestPrintWriter.println(pool.getDataSet().at(i));
    }
    samplesPrintWriter.close();
    stestPrintWriter.close();
  }

  @Test
  public void getAutomatonTest() throws IOException {
    final ProbabilisticAutomaton probabilisticAutomaton = ProbabilisticAutomaton
            .getAutomatonByFiles("perceptron/src/test/data/automaton", getActivateFunction());
    final MultiLLLogit multiLLLogit = new MultiLLLogit(12, pool.getTarget(), pool.getDataSet());
    double perplexity = probabilisticAutomaton.getPerplexity(pool.getDataSet(), multiLLLogit);
    System.out.println("Perplexity: " + perplexity);
  }

  public static FunctionC1 getActivateFunction() {
    return new FunctionC1() {
      @Override
      public double derivative(double x) {
        return 1.;
      }

      @Override
      public double call(double x) {
        return 1. / (1. + Math.exp(-x));
      }
    };
  }
}
