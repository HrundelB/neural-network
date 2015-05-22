package ru.spbsu.apmath.neuralnetwork.automaton;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
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
  private final int states;
  private HashMap<Character, Mx> weights;
  private Mx startMatrix;
  private Mx finalMatrix;
  private Vec startVec;
  private static final FastRandom RANDOM = new FastRandom();
  private CharSeq currentSeq;

  public ProbabilisticAutomaton(int states, Character[] symbols, FunctionC1 activationFunction) {
    this(states, new VecBasedMx(states, 1), new VecBasedMx(1, states),
            new HashMap<Character, Mx>(symbols.length), activationFunction);
    for (Character character : symbols) {
      Mx mx = new VecBasedMx(states, states);
      for (int j = 0; j < mx.columns(); j++) {
        Vec vec = getVecDistribution(states);
        for (int i = 0; i < mx.rows(); i++) {
          mx.set(i, j, vec.get(i));
        }
      }
      weights.put(character, mx);
    }
    Vec vec = getVecDistribution(states);
    for (int i = 0; i < states; i++) {
      startMatrix.set(i, 0, vec.get(i));
    }
    vec = getVecDistribution(states);
    for (int i = 0; i < states; i++) {
      finalMatrix.set(0, i, vec.get(i));
    }
  }

  private ProbabilisticAutomaton(int states, Mx startMatrix, Mx finalMatrix, HashMap<Character, Mx> weights, FunctionC1 activationFunction) {
    super(activationFunction);
    this.states = states;
    this.weights = weights;
    this.startMatrix = startMatrix;
    this.finalMatrix = finalMatrix;
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
    outputs = new Vec[argument.length() + 3];
    sums = new Vec[argument.length() + 2];

    outputs[0] = startVec;
    sums[0] = MxTools.multiply(startMatrix, startVec);
    outputs[1] = activationFunction.vecValue(sums[0]);

    for (int i = 0; i < argument.length(); i++) {
      sums[i + 1] = MxTools.multiply(weights.get(argument.charAt(i)), outputs[i + 1]);
      outputs[i + 2] = activationFunction.vecValue(sums[i + 1]);
    }

    sums[argument.length() + 1] = MxTools.multiply(finalMatrix, outputs[argument.length() + 1]);
    outputs[argument.length() + 2] = activationFunction.vecValue(sums[argument.length() + 1]);
    return outputs[outputs.length - 1];
  }

  @Override
  public void setLearn(CharSeq learn) {
    currentSeq = learn;
  }

  @Override
  public int depth() {
    return currentSeq.length() + 2;
  }

  @Override
  public int ydim() {
    return finalMatrix.rows();
  }

  @Override
  public Mx weights(int i) {
    if (i == 0) {
      return startMatrix;
    }
    if (i == depth() - 1) {
      return finalMatrix;
    }
    Character c = currentSeq.at(i - 1);
    Mx mx = weights.get(c);
    return mx;
  }

  @Override
  public Learnable clone() {
    HashMap<Character, Mx> hashMap = new HashMap<Character, Mx>();
    for (Character c : weights.keySet()) {
      hashMap.put(c, new VecBasedMx(weights.get(c)));
    }
    Mx startMx = new VecBasedMx(startMatrix);
    Mx finalMx = new VecBasedMx(finalMatrix);
    return new ProbabilisticAutomaton(states, startMx, finalMx, hashMap, activationFunction);
  }

  @Override
  public void save(String pathToFolder) throws IOException {
    for (Character c : weights.keySet()) {
      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, c));
      StringTools.printMx(weights.get(c), file);
    }
    File startMx = new File(String.format("%s/startMatrix.txt", pathToFolder));
    StringTools.printMx(startMatrix, startMx);
    File finalMx = new File(String.format("%s/finalMatrix.txt", pathToFolder));
    StringTools.printMx(finalMatrix, finalMx);
    File states = new File(String.format("%s/states.txt", pathToFolder));
    Mx mx = new VecBasedMx(1, 1);
    mx.set(0, 0, this.states);
    StringTools.printMx(mx, states);
  }

  public static ProbabilisticAutomaton getAutomatonByFiles(String pathToFolder, FunctionC1 activationFunction) throws IOException {
    File states = new File(String.format("%s/states.txt", pathToFolder));
    Mx mx = StringTools.readMx(states);
    File startMx = new File(String.format("%s/startMatrix.txt", pathToFolder));
    Mx startMatrix = StringTools.readMx(startMx);
    File finalMx = new File(String.format("%s/finalMatrix.txt", pathToFolder));
    Mx finalMatrix = StringTools.readMx(finalMx);
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
    return new ProbabilisticAutomaton((int) mx.get(0, 0), startMatrix, finalMatrix, hashMap, activationFunction);
  }

}
