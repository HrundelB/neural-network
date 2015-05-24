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
  private Mx[] finalMatrices;
  private static final FastRandom RANDOM = new FastRandom();
  private CharSeq currentSeq;

  public ProbabilisticAutomaton(int states, Character[] symbols, FunctionC1 activationFunction, int finalLayers) {
    this(states, new VecBasedMx(states, 1), new Mx[finalLayers],
            new HashMap<Character, Mx>(symbols.length), activationFunction);
    for (Character character : symbols) {
      Mx mx = new VecBasedMx(states, states);
      fillWithRandom(mx);
      weights.put(character, mx);
    }
    for (int i = 0; i < finalLayers; i++) {
      Mx mx;
      if (i == finalLayers - 1) {
        mx = new VecBasedMx(1, states);
      } else {
        mx = new VecBasedMx(states, states);
      }
      fillWithRandom(mx);
      finalMatrices[i] = mx;
    }
    fillWithRandom(startMatrix);
  }

  private ProbabilisticAutomaton(int states, Mx startMatrix, Mx[] finalMatrices, HashMap<Character, Mx> weights, FunctionC1 activationFunction) {
    super(activationFunction);
    this.states = states;
    this.weights = weights;
    this.startMatrix = startMatrix;
    this.finalMatrices = finalMatrices;
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
    int length = argument.length();
    int fLenght = finalMatrices.length;
    outputs = new Vec[length + fLenght + 2];
    sums = new Vec[length + fLenght + 1];

    outputs[0] = new ArrayVec(new double[]{length});
    sums[0] = MxTools.multiply(startMatrix, outputs[0]);
    outputs[1] = activationFunction.vecValue(sums[0]);

    for (int i = 0; i < length; i++) {
      sums[i + 1] = MxTools.multiply(weights.get(argument.charAt(i)), outputs[i + 1]);
      outputs[i + 2] = activationFunction.vecValue(sums[i + 1]);
    }

    for (int i = length; i < fLenght + length; i++) {
      sums[i + 1] = MxTools.multiply(finalMatrices[i - length], outputs[i + 1]);
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
    return currentSeq.length() + finalMatrices.length + 1;
  }

  @Override
  public int ydim() {
    return 1;
  }

  @Override
  public Mx weights(int i) {
    if (i == 0) {
      return startMatrix;
    }
    if (i < currentSeq.length() + 1) {
      Character c = currentSeq.at(i - 1);
      return weights.get(c);
    } else {
      return finalMatrices[i - currentSeq.length() - 1];
    }
  }

  @Override
  public Learnable clone() {
    HashMap<Character, Mx> hashMap = new HashMap<Character, Mx>();
    for (Character c : weights.keySet()) {
      hashMap.put(c, new VecBasedMx(weights.get(c)));
    }
    Mx[] fMxes = new Mx[finalMatrices.length];
    for (int i = 0; i < finalMatrices.length; i++) {
      fMxes[i] = new VecBasedMx(finalMatrices[i]);
    }
    Mx startMx = new VecBasedMx(startMatrix);
    return new ProbabilisticAutomaton(states, startMx, fMxes, hashMap, activationFunction);
  }

  @Override
  public int getComputedClass(CharSeq data) {
    return compute(data).get(0) > 0.5 ? 1 : 0;
  }

  @Override
  public void save(String pathToFolder) throws IOException {
    for (Character c : weights.keySet()) {
      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, c));
      StringTools.printMx(weights.get(c), file);
    }
    File startMx = new File(String.format("%s/startMatrix.txt", pathToFolder));
    StringTools.printMx(startMatrix, startMx);
//    File finalMx = new File(String.format("%s/finalMatrix.txt", pathToFolder));
//    StringTools.printMx(finalMatrix, finalMx);
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
    //return new ProbabilisticAutomaton((int) mx.get(0, 0), startMatrix, finalMatrix, hashMap, activationFunction);
    return null;
  }

  private void fillWithRandom(Mx mx) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        mx.set(i, j, RANDOM.nextGaussian());
      }
    }
  }
}
