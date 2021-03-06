package com.gkh.deeplearning

import org.junit.jupiter.api.Test
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

class DeepModelTest {

    @Test
    fun testTrainUsingBankNoteAuthenticationDataset() {

        val layerDims = listOf(4L,10L,1L)

        val text = this::class.java.getResource("/data_banknote_authentication.txt").readText(Charsets.UTF_8)
        val lines = text.split("\n")
        val split = (lines.size * 0.8).toInt()
        val (xTrain, yTrain) = splitTrainingData(lines, split)
        val (xPredict, yPredict) = splitTestData(lines, split)

        val parameters = DeepModel.train(
            xTrain,
            yTrain,
            layerDims,
            learningRate = 0.2,
            numIterations = 10000,
            printCost = true
        )

        val predictions = DeepModel.predict(parameters, xPredict)
        val Y = yPredict.gt(0.5).castTo(DataType.INT8)
        val YHAT = predictions.castTo(DataType.INT8)

        println("Accuracy: ${calcAccuracy(Y, YHAT)} %")
    }

    private fun splitTestData(
        lines: List<String>,
        split: Int
    ): Pair<INDArray, INDArray> {
        val xPredictArrays = mutableListOf<DoubleArray>()
        val yPredictArrays = mutableListOf<DoubleArray>()
        lines.slice(split + 1..lines.size - 1).forEach {
            val tokens = it.split(",")
            xPredictArrays.add(
                doubleArrayOf(
                    tokens.get(0).toDouble(),
                    tokens.get(1).toDouble(),
                    tokens.get(2).toDouble(),
                    tokens.get(3).toDouble()
                )
            )
            yPredictArrays.add(doubleArrayOf(tokens.get(4).toDouble()))
        }

        val xPredict = nj.create(xPredictArrays.toTypedArray()).transpose()
        val yPredict = nj.create(yPredictArrays.toTypedArray()).transpose()
        return Pair(xPredict, yPredict)
    }

    private fun splitTrainingData(
        lines: List<String>,
        split: Int
    ): Pair<INDArray, INDArray> {
        val xTrainArrays = mutableListOf<DoubleArray>()
        val yTrainArrays = mutableListOf<DoubleArray>()
        lines.slice(0..split).forEach {
            val tokens = it.split(",")
            xTrainArrays.add(
                doubleArrayOf(
                    tokens.get(0).toDouble(),
                    tokens.get(1).toDouble(),
                    tokens.get(2).toDouble(),
                    tokens.get(3).toDouble()
                )
            )
            yTrainArrays.add(doubleArrayOf(tokens.get(4).toDouble()))
        }
        val xTrain = nj.create(xTrainArrays.toTypedArray()).transpose()
        val yTrain = nj.create(yTrainArrays.toTypedArray()).transpose()
        return Pair(xTrain, yTrain)
    }

    private fun calcAccuracy(
        Y_1: INDArray,
        Y_2: INDArray
    ) = (Transforms.dot(Y_1, Y_2.transpose()).add(Transforms.dot(Y_1.rsub(1), Y_2.transpose().rsub(1)))).div(Y_1.size(1)).mul(100.0)


}