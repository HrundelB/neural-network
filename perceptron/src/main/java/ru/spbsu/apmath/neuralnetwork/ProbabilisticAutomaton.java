package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;

import java.util.HashMap;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomaton {
  private final int finalStates;
  private final int allStates;
  private HashMap<Character, Mx> weights;
  private final FastRandom random = new FastRandom();

  public ProbabilisticAutomaton(int states, int finalStates, Character[] symbols) {
    this.weights = new HashMap<Character, Mx>(symbols.length);
    this.finalStates = finalStates;
    this.allStates = 1 + states + this.finalStates;

    for (Character character : symbols) {
      Mx mx = new VecBasedMx(allStates, allStates);

      for (int j = 0; j < mx.columns() - finalStates; j++) {
        Vec vec = getVecDistibution(mx.rows() - 1);
        mx.set(0, j, 0);
        for (int i = 1; i < mx.rows(); i++) {
          mx.set(i, j, vec.get(i - 1));
        }
      }

      for (int j = mx.columns() - finalStates; j < mx.columns(); j++) {
        for (int i = 0; i < mx.rows(); i++) {
          mx.set(i, j, 0);
        }
      }

      weights.put(character, mx);
    }
  }

  private Vec getVecDistibution(int length) {
    Vec vec = new ArrayVec(length);
    double sum = 0;
    for (int i = 0; i < vec.length(); i++) {
      double val = random.nextDouble();
      vec.set(i, val);
      sum += val;
    }
    for (int i = 0; i < vec.length(); i++) {
      vec.set(i, vec.get(i) / sum);
    }
    return vec;
  }

  public Mx getWeights(Character character) {
    return weights.get(character);
  }

  public Vec calculate(CharSequence sequence) {
    Vec vec = new ArrayVec(allStates);
    vec.set(0, 1);
    for (int i = 1; i < vec.length(); i++) {
      vec.set(i, 0);
    }

    double sum = 0;

    for (int i = 0; i < sequence.length(); i++) {
      vec = MxTools.multiply(weights.get(sequence.charAt(i)), vec);
      for (int j = vec.length() - finalStates; j < vec.length(); j++) {
        double val = vec.get(j);
        sum += val;
        System.out.println(val);
      }
    }

    for (int j = 0; j < vec.length() - finalStates; j++) {
      double val = vec.get(j);
      sum += val;
    }
    System.out.println("Sum: " + sum);
    return vec;
  }
}
