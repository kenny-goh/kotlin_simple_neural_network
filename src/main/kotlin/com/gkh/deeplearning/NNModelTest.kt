package com.gkh.deeplearning

import org.junit.jupiter.api.Test
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.dot
import java.util.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue


class NNModelTest {

    @Test
    fun testLayerSize() {
        val X = Nd4j.zeros(2,100)
        val Y = Nd4j.zeros(1,100)
        val sizes = NNModel.layerSizes(X, Y)

        assertEquals( Triple(2L,4L,1L), sizes)
    }

    @Test
    fun testInitParameters() {

        val parameters = NNModel.initParameters(2, 4, 1)
        val M = arrayOf(
            doubleArrayOf(-0.0095, -0.02000),
            doubleArrayOf(-0.0097, 0.0091),
            doubleArrayOf( 0.0184, -0.0037),
            doubleArrayOf(-0.0075, 0.0124))

        assertEquals(parameters["W1"].toString(), Nd4j.create(M).toString())
        assertEquals(parameters["b1"], Nd4j.zeros(DataType.DOUBLE, 4,1))
    }

    @Test
    fun testForwardPropagation() {

        val X = Nd4j.create(arrayOf(
            doubleArrayOf(0.0,1.0)
          )).transpose()

        val parameters = NNModel.initParameters(2, 4, 1)
        val (A2, cache) = NNModel.forwardPropagation(X, parameters)
        println(cache)
    }

    @Test
    fun testComputeCost() {

        val X = Nd4j.create(arrayOf(
            doubleArrayOf(0.0,1.0)
        )).transpose()

        val Y = Nd4j.create(arrayOf(
            doubleArrayOf(1.0)
        )).transpose()

        val parameters = NNModel.initParameters(2, 4, 1)
        val (A2, cache) = NNModel.forwardPropagation(X, parameters)
        val result = NNModel.computeCost(A2, Y, parameters)
        assertTrue(result > 0.0)
    }

    @Test
    fun testBackwardPropagation() {

        val X = Nd4j.create(arrayOf(
            doubleArrayOf(0.0,1.0)
        )).transpose()

        val Y = Nd4j.create(arrayOf(
            doubleArrayOf(1.0)
        )).transpose()

        val parameters = NNModel.initParameters(2, 4, 1)
        val (A2, cache) = NNModel.forwardPropagation(X, parameters)
        val grads =
            NNModel.backwardPropagation(parameters, cache, X, Y)
        print ("dW1 = "+ grads["dW1"].toString())
        print ("db1 = "+ grads["db1"].toString())
        print ("dW2 = "+ grads["dW2"].toString())
        print ("db2 = "+ grads["db2"].toString())
    }

    @Test
    fun testUpdateParameters() {


        val X = Nd4j.create(arrayOf(
            doubleArrayOf(0.0,1.0)
        )).transpose()

        val Y = Nd4j.create(arrayOf(
            doubleArrayOf(1.0)
        )).transpose()

        var parameters = NNModel.initParameters(2, 4, 1)
        val (A2, cache) = NNModel.forwardPropagation(X, parameters)
        val grads =
            NNModel.backwardPropagation(parameters, cache, X, Y)
        parameters = NNModel.updateParameters(parameters, grads)

        print("W1 = " + parameters["W1"].toString())
        print("b1 = " + parameters["b1"].toString())
        print("W2 = " + parameters["W2"].toString())
        print("b2 = " + parameters["b2"].toString())
    }


    @Test
    fun testTrainAndPredict() {

        val random = Random()
        random.setSeed(0)

        val xArrays = mutableListOf<DoubleArray>()
        xArrays.add(doubleArrayOf(0.0,0.0));
        xArrays.add(doubleArrayOf(0.0,0.0));
        xArrays.add(doubleArrayOf(0.0,0.0));
        xArrays.add(doubleArrayOf(0.0,0.0));
        xArrays.add(doubleArrayOf(0.0,0.0));
        xArrays.add(doubleArrayOf(1.0,1.0));
        xArrays.add(doubleArrayOf(1.0,1.0));
        xArrays.add(doubleArrayOf(1.0,1.0));
        xArrays.add(doubleArrayOf(1.0,1.0));
        xArrays.add(doubleArrayOf(1.0,1.0));
        xArrays.add(doubleArrayOf(2.0,1.0));
        xArrays.add(doubleArrayOf(2.0,1.0));
        xArrays.add(doubleArrayOf(2.0,1.0));
        xArrays.add(doubleArrayOf(2.0,1.0));
        xArrays.add(doubleArrayOf(2.0,1.0));
        val X = Nd4j.create(xArrays.toTypedArray()).transpose()

        val yArrays = mutableListOf<DoubleArray>()
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 1.0 ))
        yArrays.add(doubleArrayOf( 1.0 ))
        yArrays.add(doubleArrayOf( 1.0 ))
        yArrays.add(doubleArrayOf( 1.0 ))
        yArrays.add(doubleArrayOf( 1.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        yArrays.add(doubleArrayOf( 0.0 ))
        val Y = Nd4j.create(yArrays.toTypedArray()).transpose()

        val parameters = NNModel.nn_model(
            X,
            Y,
            4,
            num_iterations = 2000,
            print_cost = true
        )

        val predictions = NNModel.predict(parameters, X)
        val Y_INT = Y.gt(0.5).castTo(DataType.INT8)
        val Y_HAT_INT = predictions.castTo(DataType.INT8)

        println("Accuracy: ${calcAccuracy(Y_INT, Y_HAT_INT)} %")
    }

    @Test
    fun testBankNoteDataset() {
        val text = this::class.java.getResource("/data_banknote_authentication.txt").readText(Charsets.UTF_8)
        val lines = text.split("\n")
        val split = (lines.size * 0.8).toInt()
        val xTrainArrays = mutableListOf<DoubleArray>()
        val yTrainArrays = mutableListOf<DoubleArray>()
        lines.slice(0..split).forEach {
            val tokens= it.split(",")
            xTrainArrays.add(doubleArrayOf(
                tokens.get(0).toDouble(),
                tokens.get(1).toDouble(),
                tokens.get(2).toDouble(),
                tokens.get(3).toDouble()))
            yTrainArrays.add(doubleArrayOf(tokens.get(4).toDouble()))
        }
        val xTrain = Nd4j.create(xTrainArrays.toTypedArray()).transpose()
        val yTrain = Nd4j.create(yTrainArrays.toTypedArray()).transpose()

        val xPredictArrays = mutableListOf<DoubleArray>()
        val yPredictArrays = mutableListOf<DoubleArray>()
        lines.slice(split+1..lines.size-1).forEach {
            val tokens= it.split(",")
            xPredictArrays.add(doubleArrayOf(
                tokens.get(0).toDouble(),
                tokens.get(1).toDouble(),
                tokens.get(2).toDouble(),
                tokens.get(3).toDouble()))
            yPredictArrays.add(doubleArrayOf(tokens.get(4).toDouble()))
        }

        val xPredict = Nd4j.create(xPredictArrays.toTypedArray()).transpose()
        val yPredict = Nd4j.create(yPredictArrays.toTypedArray()).transpose()

        val parameters = NNModel.nn_model(
            xTrain,
            yTrain,
            4,
            num_iterations = 10000,
            print_cost = true
        )

        val predictions = NNModel.predict(parameters, xPredict)
        val Y_INT = yPredict.gt(0.5).castTo(DataType.INT8)
        val Y_HAT_INT = predictions.castTo(DataType.INT8)

        println("Accuracy: ${calcAccuracy(Y_INT, Y_HAT_INT)} %")

    }

    private fun calcAccuracy(
        Y_1: INDArray,
        Y_2: INDArray
    ) = (dot(Y_1, Y_2.transpose()).add(dot(Y_1.rsub(1), Y_2.transpose().rsub(1)))).div(Y_1.size(1)).mul(100.0)


}