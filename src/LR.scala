import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.MLUtils

object LR {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("LR").setMaster("local")
    val sc = new SparkContext(conf)

    val raw = MLUtils.loadLibSVMFile(sc, "file:///home/zl/matrix.txt")
    val parsed = raw.cache()
    println(parsed.count())
    val model = new LinearRegressionWithSGD()
    model.optimizer.setNumIterations(1000)
    model.setIntercept(true)

    val tstart = System.currentTimeMillis()
    val res = model.run(parsed)
    val tend = System.currentTimeMillis()

    println("Model training time: " + (tend - tstart) / 1000.0 + " secs")

    println(res.intercept)
    println(res.weights)

    val pred = res.predict(parsed.map(_.features))


    val r = Statistics.corr(pred, parsed.map(_.label))
    println("R-square = " + r * r)


  }
}
