package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.func.Processor;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqArray;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.data.set.DataSet;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 28.10.2014
 * Time: 20:58
 */
public class StringTools {

  public static String printMx(Mx mx) throws IOException {
    StringWriter stringWriter = new StringWriter();
    printMx(mx, stringWriter);
    return stringWriter.toString();
  }

  public static void printMx(Mx mx, File file) throws IOException {
    FileWriter fileWriter = new FileWriter(file);
    printMx(mx, fileWriter);
  }

  public static void printMx(Mx mx, Writer writer) throws IOException {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        sb.append(mx.get(i, j)).append(' ');
      }
      sb.delete(sb.length() - 1, sb.length());
      sb.append(System.lineSeparator());
    }
    sb.delete(sb.length() - 1, sb.length());
    writer.write(sb.toString());
    writer.close();
  }

  public static Mx readMx(File file) throws IOException {
    return readMx(new FileReader(file));
  }

  public static Mx readMx(String s) throws IOException {
    return readMx(new StringReader(s));
  }

  public static Mx readMx(Reader reader) throws IOException {
    List<Vec> rows = new ArrayList<Vec>();
    BufferedReader bufferedReader = new BufferedReader(reader);
    String line;
    while ((line = bufferedReader.readLine()) != null) {
      VecBuilder vecBuilder = new VecBuilder();
      String[] elements = line.split(" ");
      for (String s : elements)
        vecBuilder.append(Double.parseDouble(s));
      rows.add(vecBuilder.build());
    }
    bufferedReader.close();
    return new RowsVecArrayMx(rows.toArray(new Vec[]{}));
  }

  public static String printVec(Vec vec) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < vec.dim(); i++) {
      sb.append(vec.get(i)).append(' ');
    }
    sb.delete(sb.length() - 1, sb.length());
    return sb.toString();
  }

  public static Pair<List<CharSeq>, List<Double>> loadTrainTxt(final String file) throws IOException {
    return loadTrainTxt(file.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(new FileInputStream(file))) : new FileReader(file));
  }

  public static Pair<List<CharSeq>, List<Double>> loadTrainTxt(final Reader in) throws IOException {
    System.out.println("Start loading...");
    final List<Double> target = new ArrayList<Double>();
    final List<CharSeq> data = new ArrayList<CharSeq>();
    Processor<CharSequence> processor = new Processor<CharSequence>() {
      private int index = 0;

      @Override
      public void process(CharSequence arg) {
        index++;
        try {
          final CharSequence[] parts = CharSeqTools.split(arg, '\t');
          final CharSequence[] numbers = CharSeqTools.split(parts[1], ':');
          CharSeq charSeq = new CharSeqArray(getArray(parts[0]));
          Double d = Double.parseDouble(numbers[1].toString());
          data.add(charSeq);
          target.add(d);
        } catch (Exception e) {
          System.out.println(String.format("Failed to read line %s: %s", index, arg));
        }
        if (index % 1000 == 0) {
          System.out.print(String.format("line: %s\r", index));
        }
      }
    };

    try {
      CharSeqTools.processLines(in, processor);
    } catch (RuntimeException e) {
    }
    System.out.println(String.format("Finish loading, total data: %s, target: %s", data.size(), target.size()));
    return new Pair<List<CharSeq>, List<Double>>(data, target);
  }

  public static char[] getArray(CharSequence sequence) {
    char[] chars = new char[sequence.length()];
    for (int i = 0; i < chars.length; i++) {
      chars[i] = sequence.charAt(i);
    }
    return chars;
  }

  public static Character[] findCharacters(DataSet<CharSeq> dataSet) throws IOException {
    Set<Character> set = new HashSet<Character>();
    for (int i = 0; i < dataSet.length(); i++) {
      CharSeq charSeq = dataSet.at(i);
      for (int j = 0; j < charSeq.length(); j++) {
        set.add(charSeq.at(j));
      }
    }
    System.out.println(set);
    return set.toArray(new Character[]{});
  }
}
