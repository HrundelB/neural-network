package ru.spbsu.apmath.neuralnetwork;

import cc.mallet.fst.*;
import cc.mallet.optimize.Optimizable;
import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import org.junit.Test;

/**
 * Created by afonin.s on 22.05.2015.
 */
public class CRFTest {

  @Test
  public void crfTest() throws Exception {
    //SimpleTagger.main(new String[]{"--train", "true", "--model-file", "nouncrf",  "src/test/data/sample.txt"});
    SimpleTagger.main(new String[]{ "--model-file", "src/test/data/nouncrf",  "src/test/data/stest.txt"});
  }
}
