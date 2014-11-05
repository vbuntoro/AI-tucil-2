/*
 * Copyright (C) 2014 Michael
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package tucil2;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Michael
 */
public class Tucil2 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        Instances dataset = DataSource.read(args[0]);
        dataset.setClassIndex(dataset.numAttributes() - 1);
        if (args[1].equals("J48")) {
            String[] options = new String[4];
            options[0] = "-C";
            options[1] = "0.25";
            options[2] = "-M";
            options[3] = "2";
            J48 klas = new J48();
            klas.setOptions(options);
            klas.buildClassifier(dataset);
            System.out.println(klas.toSummaryString());
            //Evaluation Build
            Evaluation eval = new Evaluation(dataset);
            if (args[2].equals("cross")) {
                eval.crossValidateModel(klas, dataset, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else if (args[2].equals("fullset")) {
                Classifier cls = new J48();
                cls.buildClassifier(dataset);
                eval.evaluateModel(cls, dataset);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else {
                System.err.println("args 2 must be either cross or fullset only");
                System.exit(1);
            }
        } else if (args[1].equals("ibk")) {
            String[] options = new String[6];
            options[0] = "-K";
            options[1] = "1";
            options[2] = "-W";
            options[3] = "0";
            options[4] = "-A";
            options[5] = "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"";
            IBk klas = new IBk();
            klas.setOptions(options);
            klas.buildClassifier(dataset);
            //Evaluation Build
            Evaluation eval = new Evaluation(dataset);
            if (args[2].equals("cross")) {
                eval.crossValidateModel(klas, dataset, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else if (args[2].equals("fullset")) {
                Classifier cls = new IBk();
                cls.buildClassifier(dataset);
                eval.evaluateModel(cls, dataset);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else {
                System.err.println("args 2 must be either cross or fullset only");
                System.exit(1);
            }
        } else if (args[1].equals("perceptron")) {
            String[] options = new String[15];
            options[0] = "-L";
            options[1] = "0.3";
            options[2] = "-M";
            options[4] = "0.2";
            options[5] = "-N";
            options[6] = "500";
            options[7] = "-V";
            options[8] = "0";
            options[9] = "-S";
            options[10] = "0";
            options[11] = "-E";
            options[12] = "20";
            options[13] = "-H";
            options[14] = "a";
            MultilayerPerceptron klas = new MultilayerPerceptron();
            //klas.setOptions(options);
            klas.buildClassifier(dataset);
            //Evaluation Build
            Evaluation eval = new Evaluation(dataset);
            if (args[2].equals("cross")) {
                eval.crossValidateModel(klas, dataset, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else if (args[2].equals("fullset")) {
                Classifier cls = new MultilayerPerceptron();
                cls.buildClassifier(dataset);
                eval.evaluateModel(cls, dataset);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else {
                System.err.println("args 2 must be either cross or fullset only");
                System.exit(1);
            }
        } else if (args[1].equals("bayes")) {
            NaiveBayes klas = new NaiveBayes();
            klas.buildClassifier(dataset);
            //Evaluation Build
            Evaluation eval = new Evaluation(dataset);
            if (args[2].equals("cross")) {
                eval.crossValidateModel(klas, dataset, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else if (args[2].equals("fullset")) {
                Classifier cls = new NaiveBayes();
                cls.buildClassifier(dataset);
                eval.evaluateModel(cls, dataset);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else {
                System.err.println("args 2 must be either cross or fullset only");
                System.exit(1);
            }
        } else {
            System.err.println("Valid args 1 are J48 / ibk / perceptron / bayes");
            System.exit(1);
        }
    }

}
