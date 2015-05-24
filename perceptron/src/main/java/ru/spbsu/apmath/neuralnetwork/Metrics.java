package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;
import ru.spbsu.apmath.neuralnetwork.perceptron.LLLogit;

import java.util.HashMap;

import static ru.spbsu.apmath.neuralnetwork.Metrics.Decision.*;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 24.05.2015
 * Time: 20:18
 */
public class Metrics {

  public static double getPerplexity(DataSet dataSet, double logLikelihood) {
    double d = 1.0 / dataSet.length();
    return Math.exp(-d * logLikelihood);
  }

  public static <L extends Seq> void printMetrics(Learnable<L> learnable, DataSet<L> dataSet, LLLogit logit, int categories) {
    HashMap<Integer, HashMap<Decision, Double>> decisions = new HashMap<>(categories);
    for (int i = 0; i < categories; i++) {
      HashMap<Decision, Double> hashMap = new HashMap<>(values().length);
      for (Decision decision : values()) {
        hashMap.put(decision, 0.);
      }
      decisions.put(i, hashMap);
    }
    double accuracy = 0;
    for (int i = 0; i < dataSet.length(); i++) {
      int answer = learnable.getComputedClass(dataSet.at(i));
      int target = logit.getTargetClass(i);
        if (target == answer) {
          accuracy++;
          increment(decisions, answer, TRUE_POSITIVE);
          increment(decisions, target, TRUE_NEGATIVE);
        } else {
          increment(decisions, answer, FALSE_POSITIVE);
          increment(decisions, target, FALSE_NEGATIVE);
        }
    }
    System.out.println(String.format("Accuracy: %s", accuracy / dataSet.length()));
    for (int i = 0; i < categories; i++) {
      HashMap<Decision, Double> map = decisions.get(i);
      double precision = map.get(TRUE_POSITIVE) / (map.get(TRUE_POSITIVE) + map.get(FALSE_POSITIVE));
      double recall = map.get(TRUE_POSITIVE) / (map.get(TRUE_POSITIVE) + map.get(FALSE_NEGATIVE));
      System.out.println(String.format("[%s] Precision: %s, Recall: %s", i, precision, recall));
    }
  }

  private static void increment (HashMap<Integer, HashMap<Decision, Double>> decisions, int c, Decision d) {
    Double n = decisions.get(c).get(d) + 1;
    decisions.get(c).put(d, n);
  }

  public enum Decision {
    TRUE_POSITIVE,
    TRUE_NEGATIVE,
    FALSE_POSITIVE,
    FALSE_NEGATIVE
  }
}
