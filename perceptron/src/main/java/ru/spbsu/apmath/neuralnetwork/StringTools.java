package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

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
        sb.append(mx.get(i,j)).append(' ');
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
      for (String s: elements)
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
}
