import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object SVM {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("LR").setMaster("local[*]")
    val sc = new SparkContext(conf)


    val raw = MLUtils.loadLibSVMFile(sc, "file:///home/zl/matrix.txt")

    val splits = raw.randomSplit(Array(0.9, 0.1), seed = 11L)

    val training = splits(0).cache()
    val test = splits(1)

    val numIterations = 1000
    val model = SVMWithSGD.train(training, numIterations)

    model.clearThreshold()
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      val s = if (score > 0)
        1.0
      else
        -1.0
      (s, point.label)
    }

    scoreAndLabels.foreach(println)
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(scoreAndLabels)

    println("f : " + metrics.fMeasure(1))
    println("pr : " + metrics.precision(1))
    println("re : " + metrics.recall(1))


  }
}
