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
        if(args[1].equals("J48")) {
            String[] options = new String[4];
            options[0] = "-C";
            options[1] = "0.25";
            options[2] = "-M";
            options[3] = "2";
            J48 tree = new J48();
            tree.setOptions(options);
            tree.buildClassifier(dataset);
            System.out.println(tree.toSummaryString());
            //Evaluation Build
            Evaluation eval = new Evaluation(dataset);
            if(args[2].equals("cross")) {
                eval.crossValidateModel(tree, dataset, 10, new Random(1));
                System.out.println(eval.toSummaryString("\nResults\n\n", false));  
            } else if(args[2].equals("fullset")) {
                Classifier cls = new J48();
                cls.buildClassifier(dataset);
                eval.evaluateModel(cls, dataset);
                System.out.println(eval.toSummaryString("\nResults\n\n", false));
            } else {
                System.err.println("args 2 must be either cross or fullset only");
                System.exit(1);
            }
        }
        else if(args[1].equals("ibk")) {
                String[] options = new String[4];
                options[0] = "-C";
                options[1] = "0.25";
                options[2] = "-M";
                options[3] = "2";
                J48 tree = new J48();
                tree.setOptions(options);
                tree.buildClassifier(dataset);
                System.out.println(tree.toSummaryString());
                //Evaluation Build
                Evaluation eval = new Evaluation(dataset);
                if(args[2].equals("cross")) {
                    eval.crossValidateModel(tree, dataset, 10, new Random(1));
                    System.out.println(eval.toSummaryString("\nResults\n\n", false));  
                } else if(args[2].equals("fullset")) {
                    Classifier cls = new J48();
                    cls.buildClassifier(dataset);
                    eval.evaluateModel(cls, dataset);
                    System.out.println(eval.toSummaryString("\nResults\n\n", false));
                } else {
                    System.err.println("args 2 must be either cross or fullset only");
                    System.exit(1);
                }
    }
        
    }
    
    
}
