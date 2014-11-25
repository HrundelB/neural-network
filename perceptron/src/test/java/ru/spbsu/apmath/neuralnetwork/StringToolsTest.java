package ru.spbsu.apmath.neuralnetwork;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static org.hamcrest.MatcherAssert.assertThat;
import static ru.spbsu.apmath.neuralnetwork.MxEqualsMatcher.mxEqualsTo;
import static ru.spbsu.apmath.neuralnetwork.StringTools.printMx;
import static ru.spbsu.apmath.neuralnetwork.StringTools.readMx;

/**
 * Created by IntelliJ IDEA.
 * User: Афонин Сергей (hrundelb@yandex.ru)
 * Date: 28.10.2014
 * Time: 21:18
 */
public class StringToolsTest {

  @Test
  public void printMxTest() throws IOException {
    Mx original = getMx();
    String s = printMx(original);
    Mx newMx = readMx(s);
    assertThat(newMx, mxEqualsTo(original));
  }

  private Mx getMx() {
    Vec vec1 = new ArrayVec(2, 5, 8);
    Vec vec2 = new ArrayVec(1, 51, 81);
    Vec vec3 = new ArrayVec(32, 52, 18);
    Vec vec4 = new ArrayVec(200, 0.5858585855, 8.0000000000123);
    return new RowsVecArrayMx(new Vec[]{vec1, vec2, vec3, vec4});
  }

  @Test
  public void writeMxToFileTest() throws IOException {
    Mx original = getMx();
    File file = new File("perceptron/src/test/data/perceptron/testmatrix1.txt");
    printMx(original, file);
    Mx newMx = readMx(file);
    assertThat(newMx, mxEqualsTo(original));
  }
}
