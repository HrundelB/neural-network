package ru.spbsu.apmath.neuralnetwork.automaton;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeq;
import ru.spbsu.apmath.neuralnetwork.Learnable;
import ru.spbsu.apmath.neuralnetwork.backpropagation.Function;

import java.io.IOException;
import java.util.HashMap;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomaton extends Learnable<CharSeq> {
  private final int finalStates;
  private final int allStates;
  private HashMap<Character, Mx> weights;
  private final FastRandom random = new FastRandom();
  private CharSeq currentSeq;
  private Vec[] outputs;
  private Vec[] sums;
  private final Function activationFunction;

  public ProbabilisticAutomaton(int states, int finalStates, Character[] symbols, Function activationFunction) {
    this.weights = new HashMap<Character, Mx>(symbols.length);
    this.finalStates = finalStates;
    this.allStates = 1 + states + this.finalStates;
    this.activationFunction = activationFunction;

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

  private ProbabilisticAutomaton(int allStates, int finalStates, HashMap<Character, Mx> weights, Function activationFunction) {
    this.allStates = allStates;
    this.finalStates = finalStates;
    this.weights = weights;
    this.activationFunction = activationFunction;
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

  @Override
  public Vec compute(CharSeq argument) {
    outputs = new Vec[argument.length() + 1];
    sums = new Vec[argument.length()];

    outputs[0] = new ArrayVec(allStates);
    outputs[0].set(0, 1);
    for (int i = 1; i < outputs[0].length(); i++) {
      outputs[0].set(i, 0);
    }

    for (int i = 0; i < argument.length(); i++) {
      sums[i] = MxTools.multiply(weights.get(argument.charAt(i)), outputs[i]);
      outputs[i + 1] = activationFunction.vecValue(sums[i]);
    }
    return outputs[outputs.length - 1];
  }

  @Override
  public void setLearn(CharSeq learn) {
    currentSeq = learn;
  }

  @Override
  public int depth() {
    return currentSeq.length();
  }

  @Override
  public int ydim() {
    return allStates;
  }

  @Override
  public Mx weights(int i) {
    Character c = currentSeq.at(i);
    return weights.get(c);
  }

  @Override
  public Vec getSum(int i) {
    return sums[i];
  }

  @Override
  public Vec getOutput(int i) {
    return outputs[i + 1];
  }

  @Override
  public Learnable clone() {
    HashMap<Character, Mx> hashMap = new HashMap<Character, Mx>();
    for (Character c : weights.keySet()) {
      hashMap.put(c, new VecBasedMx(weights.get(c)));
    }
    return new ProbabilisticAutomaton(allStates, finalStates, hashMap, activationFunction);
  }

  @Override
  public void save(String pathToFolder) throws IOException {
    throw new IOException("null");
  }
}
