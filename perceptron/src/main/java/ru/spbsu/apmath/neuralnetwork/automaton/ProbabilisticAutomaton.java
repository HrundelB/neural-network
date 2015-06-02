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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by afonin.s on 23.03.2015.
 */
public class ProbabilisticAutomaton extends Learnable<CharSeq> {
  private final int states;
  private HashMap<Character, List<Mx>> weights;
  private Mx startMatrix;
  private Mx[] finalMatrices;
  private static final FastRandom RANDOM = new FastRandom();
  private CharSeq currentSeq;

  public ProbabilisticAutomaton(int states, Character[] symbols, FunctionC1 activationFunction, int finalLayers) {
    this(states, new VecBasedMx(states, 1), new Mx[finalLayers],
            new HashMap<Character, List<Mx>>(symbols.length), activationFunction);
    for (Character character : symbols) {
      List<Mx> list = new ArrayList<Mx>();
      Mx mx = new VecBasedMx(states, states);
      fillWithRandom(mx);
      list.add(mx);
      weights.put(character, list);
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

  private ProbabilisticAutomaton(int states, Mx startMatrix, Mx[] finalMatrices, HashMap<Character, List<Mx>> weights, FunctionC1 activationFunction) {
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
    setLearn(argument);
    outputs = new Vec[depth() + 1];
    sums = new Vec[depth()];

    outputs[0] = new ArrayVec(new double[]{depth()});

    for (int i = 0; i < depth(); i++) {
      sums[i] = MxTools.multiply(weights(i), outputs[i]);
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
      int n = 0;
      for (int j = 0; j < i - 1; j++) {
        if (currentSeq.charAt(j) == c) {
          n++;
        }
      }
      List<Mx> list = weights.get(c);
      if (list.size() <= n) {
        for (int j = 0; j < n - list.size() + 1; j++) {
          list.add(new VecBasedMx(list.get(0)));
        }
      }
      weights.put(c, list);
      return weights.get(c).get(n);
    } else {
      return finalMatrices[i - currentSeq.length() - 1];
    }
  }

  @Override
  public Learnable clone() {
    HashMap<Character, List<Mx>> hashMap = new HashMap<Character, List<Mx>>();
    for (Character c : weights.keySet()) {
      List<Mx> mxes = weights.get(c);
      List<Mx> newMxes = new ArrayList<>(mxes.size());
      for (int i = 0; i < mxes.size(); i++) {
        newMxes.add(new VecBasedMx(mxes.get(i)));
      }
      hashMap.put(c, newMxes);
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
//    for (Character c : weights.keySet()) {
//      File file = new File(String.format("%s/matrix%s.txt", pathToFolder, c));
//      StringTools.printMx(weights.get(c), file);
//    }
//    File startMx = new File(String.format("%s/startMatrix.txt", pathToFolder));
//    StringTools.printMx(startMatrix, startMx);
////    File finalMx = new File(String.format("%s/finalMatrix.txt", pathToFolder));
////    StringTools.printMx(finalMatrix, finalMx);
//    File states = new File(String.format("%s/states.txt", pathToFolder));
//    Mx mx = new VecBasedMx(1, 1);
//    mx.set(0, 0, this.states);
//    StringTools.printMx(mx, states);
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
