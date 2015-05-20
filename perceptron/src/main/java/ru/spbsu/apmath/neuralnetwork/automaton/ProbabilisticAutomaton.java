package ru.spbsu.apmath.neuralnetwork.automaton;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeq;
import ru.spbsu.apmath.neuralnetwork.Learnable;
import ru.spbsu.apmath.neuralnetwork.StringTools;
import ru.spbsu.apmath.neuralnetwork.backpropagation.FunctionC1;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomaton extends Learnable<CharSeq> {
  private final int finalStates;   // real final states count
  private final int allStates;     // states + finalStates - 1
  private HashMap<Character, Mx> weights;
  private Mx startMatrix;
  private Vec startVec;
  private static final FastRandom RANDOM = new FastRandom();
  private CharSeq currentSeq;

  public ProbabilisticAutomaton(int states, int finalStates, Character[] symbols, FunctionC1 activationFunction) {
    this(states + finalStates - 1, finalStates, new HashMap<Character, Mx>(symbols.length), activationFunction);
    for (Character character : symbols) {
      Mx mx = new VecBasedMx(allStates, allStates);
      for (int j = 0; j < mx.columns() - finalStates; j++) {
        Vec vec = getVecDistribution(mx.rows());
        for (int i = 0; i < mx.rows(); i++) {
          mx.set(i, j, vec.get(i));
        }
      }
      weights.put(character, mx);
    }
  }

  private ProbabilisticAutomaton(int allStates, int finalStates, HashMap<Character, Mx> weights, FunctionC1 activationFunction) {
    this(allStates, finalStates, new ColsVecArrayMx(new Vec[]{getVecDistribution(allStates)}), weights, activationFunction);
  }

  private ProbabilisticAutomaton(int allStates, int finalStates, Mx startMatrix, HashMap<Character, Mx> weights, FunctionC1 activationFunction) {
    super(activationFunction);
    this.allStates = allStates;
    this.finalStates = finalStates;
    this.weights = weights;
    this.startMatrix = startMatrix;
    this.startVec = new ArrayVec(new double[]{1});
  }

  private static Vec getVecDistribution(int length) {
    Vec vec = new ArrayVec(length);
    double sum = 0;
    for (int i = 0; i < vec.length(); i++) {
      double val = RANDOM.nextDouble();
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
    outputs = new Vec[argument.length() + 2];
    sums = new Vec[argument.length() + 1];

    outputs[0] = startVec;
    sums[0] = MxTools.multiply(startMatrix, startVec);
    outputs[1] = activationFunction.vecValue(sums[0]);

    for (int i = 0; i < argument.length(); i++) {
      sums[i + 1] = MxTools.multiply(weights.get(argument.charAt(i)), outputs[i + 1]);
      outputs[i + 2] = activationFunction.vecValue(sums[i + 1]);
    }
    return outputs[outputs.length - 1];
  }

  @Override
  public void setLearn(CharSeq learn) {
    currentSeq = learn;
  }

  @Override
  public int depth() {
    return currentSeq.length() + 1;
  }

  @Override
  public int ydim() {
    return allStates;
  }

  @Override
  public Mx weights(int i) {
    if (i == 0) {
      return startMatrix;
    }
    Character c = currentSeq.at(i - 1);
    Mx mx = weights.get(c);
    for (int j = mx.columns() - finalStates + 1; j < mx.columns(); j++) {
      for (int k = 0; k < mx.rows(); k++) {
        mx.set(k, j, 0);
      }
    }
    return mx;
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
    for (Character c : weights.keySet()) {
      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, c));
      StringTools.printMx(weights.get(c), file);
    }
    File startMx = new File(String.format("%s/startMatrix.txt", pathToFolder));
    StringTools.printMx(startMatrix, startMx);
    File states = new File(String.format("%s/states.txt", pathToFolder));
    Mx mx = new VecBasedMx(1, 2);
    mx.set(0, 0, allStates);
    mx.set(0, 1, finalStates);
    StringTools.printMx(mx, states);
  }

  public static ProbabilisticAutomaton getAutomatonByFiles(String pathToFolder, FunctionC1 activationFunction) throws IOException {
    File states = new File(String.format("%s/states.txt", pathToFolder));
    Mx mx = StringTools.readMx(states);
    File startMx = new File(String.format("%s/startMatrix.txt", pathToFolder));
    Mx startMX = StringTools.readMx(startMx);
    HashMap<Character, Mx> hashMap = new HashMap<Character, Mx>();
    Pattern p = Pattern.compile("matrix(\\w).txt");
    File folder = new File(pathToFolder);
    for (File f : folder.listFiles()) {
      if (f.isFile()) {
        Matcher m = p.matcher(f.getName());
        if (m.find()) {
          hashMap.put(m.group(1).charAt(0), StringTools.readMx(f));
        }
      }
    }
    return new ProbabilisticAutomaton((int) mx.get(0, 0), (int) mx.get(0, 1), startMX, hashMap, activationFunction);
  }
}
