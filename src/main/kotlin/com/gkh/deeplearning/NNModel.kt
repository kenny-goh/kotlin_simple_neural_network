package com.gkh.deeplearning

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.*

/**
 * Kotlin implementation of a 2-layer neural network based on course lessons from
 * deeplearning.ai from Coursera on Neural Networks and Deep Learning.
 */
class NNModel {

    companion object {
        /**
         * Arguments:
         *  X : input dataset of shape (input size, number of examples)
         *  Y : labels of shape (output size, number of examples)
         *
         * Returns:
         *  n_x : the size of the input layer
         *  n_h : the size of the hidden layer
         *  n_y : the size of the output layer
         */
        fun layerSizes(X: INDArray, Y: INDArray): Triple<Long, Long, Long> {
            val n_x = X.size(0)
            val n_h = 4L
            val n_y = Y.size(0)
            return Triple(n_x, n_h, n_y)
        }

        /**
         * Argument:
         *  n_x : size of the input layer
         *  n_h : size of the hidden layer
         *  n_y : size of the output layer
         *
         * Returns:
         *  params :  map containing parameters:
         *  W1 : weight matrix of shape (n_h, n_x)
         *  b1 : bias vector of shape (n_h, 1)
         *  W2 : weight matrix of shape (n_y, n_h)
         *  b2 : bias vector of shape (n_y, 1)
         *
         */
        fun initParameters(n_x: Long, n_h: Long, n_y: Long): MutableMap<String, INDArray> {

            // For testing purpose
            Nd4j.getRandom().setSeed(2)

            val W1 = Nd4j.randn(DataType.DOUBLE, n_h, n_x).mul(0.01)
            val b1 = Nd4j.zeros(DataType.DOUBLE, n_h, 1)
            val W2 = Nd4j.randn(DataType.DOUBLE, n_y, n_h).mul(0.01)
            val b2 = Nd4j.zeros(DataType.DOUBLE, n_y, 1)

            assert(W1.shape().toList() == listOf(n_h, n_x))
            assert(b1.shape().toList() == listOf(n_h, 1))
            assert(W2.shape().toList() == listOf(n_y, n_h))
            assert(b2.shape().toList() == listOf(n_y, 1))

            val parameters = mutableMapOf(
                "W1" to W1,
                "b1" to b1,
                "W2" to W2,
                "b2" to b2
            )

            return parameters

        }

        /**
         * Argument:
         *  X : input data of size (n_x, m)
         *  parameters : map containing  parameters (output of initialization function)
         *
         * Returns:
         *  A2 : The sigmoid output of the second activation
         *  cache : a map containing "Z1", "A1", "Z2" and "A2"
         */
        fun forwardPropagation(X: INDArray, parameters: Map<String, INDArray>):
                Pair<INDArray, MutableMap<String, INDArray>> {

            // Retrieve each parameter from the map "parameters"
            val W1 = parameters["W1"] ?: error("W1 is null")
            val b1 = parameters["b1"] ?: error("b1 is null")
            val W2 = parameters["W2"] ?: error("W2 is null")
            val b2 = parameters["b2"] ?: error("b2 is null")

            // Implement Forward Propagation to calculate A2 (probabilities)
            val Z1 = (W1.mmul(X)).add(b1)
            val A1 = tanh(Z1)
            val Z2 = (W2.mmul(A1)).add(b2)
            val A2 = sigmoid(Z2)

            assert(A2.shape().toList() == listOf(1L, X.size(1)))

            val cache = mutableMapOf(
                "Z1" to Z1,
                "A1" to A1,
                "Z2" to Z2,
                "A2" to A2
            )

            return Pair(A2, cache)
        }

        /**
         * Computes the cross-entropy cost given in equation (13)
         *
         * Arguments:
         *  A2 : The sigmoid output of the second activation, of shape (1, number of examples)
         *  Y : "true" labels vector of shape (1, number of examples)
         *
         * Returns:
         *  cost : cross-entropy cost given equation (13)
         */
        fun computeCost(A2: INDArray, Y: INDArray, parameters: Map<String, INDArray>): Double {
            // number of example
            val m = Y.size(1)
            // Compute the cross-entropy cost
            val a = (log(A2)).mul(Y)
            val b = (Y.rsub(1)).mul(log(A2.rsub(1)))
            val logprobs = a.add(b)
            val cost = (logprobs.sum(true, 1)).mul(-1.0 / m)
            // makes sure cost is the dimension we expect.
            val result = cost.getDouble(0)
            // E.g., turns [[17]] into 17
            assert(result is Double)


            return result

        }

        /**
         * Implement the backward propagation using the instructions above.
         *
         * Arguments:
         *  parameters :  map containing our parameters
         *  cache : a map containing "Z1", "A1", "Z2" and "A2".
         *  X : input data of shape (2, number of examples)
         *  Y : "true" labels vector of shape (1, number of examples)
         *
         * Returns:
         *  grads : map containing gradients with respect to different parameters
         *
         */
        fun backwardPropagation(
            parameters: Map<String, INDArray>,
            cache: Map<String, INDArray>,
            X: INDArray,
            Y: INDArray
        ):
                MutableMap<String, INDArray> {

            val m = X.size(1)

            // First, retrieve W1 and W2 from the map "parameters".
            val W1 = parameters["W1"] ?: error("W1 is null")
            val W2 = parameters["W2"] ?: error("W2 is null")

            // Retrieve also A1 and A2 from map "cache".
            val A1 = cache["A1"] ?: error("A1 is null")
            val A2 = cache["A2"] ?: error("A2 is null")

            // Backward propagation: calculate dW1, db1, dW2, db2.
            val dZ2 = A2.sub(Y)
            val dW2 = (dZ2.mmul(A1.transpose())).mul(1.0 / m)
            val db2 = (dZ2.sum(true, 1)).mul(1.0 / m)
            val dZ1 = (W2.transpose().mmul(dZ2)).mul((pow(A1, 2)).rsub(1))
            val dW1 = (dZ1.mmul(X.transpose())).mul(1.0 / m)
            val db1 = (dZ1.sum(true, 1)).mul(1.0 / m)

            return mutableMapOf(
                "dW1" to dW1,
                "db1" to db1,
                "dW2" to dW2,
                "db2" to db2
            )

        }


        /**
         * Updates parameters using the gradient descent update rule given above
         *
         * Arguments:
         *  parameters : map containing  parameters
         *  grads : map containing  gradients
         * Returns:
         *  parameters : map containing updated parameters
         */
        fun updateParameters(
            parameters: Map<String, INDArray>,
            grads: Map<String, INDArray>,
            learningRate: Float = 1.2f
        ): MutableMap<String, INDArray> {

            // Retrieve each parameter from the map "parameters"
            var W1 = parameters["W1"] ?: error("W1 is null")
            var b1 = parameters["b1"] ?: error("b1 is null")
            var W2 = parameters["W2"] ?: error("W2 is null")
            var b2 = parameters["b2"] ?: error("b2 is null")

            // Retrieve each gradient from the map "grads"
            val dW1 = grads["dW1"] ?: error("dW1 is null")
            val db1 = grads["db1"] ?: error("db1 is null")
            val dW2 = grads["dW2"] ?: error("dW2 is null")
            val db2 = grads["db2"] ?: error("db2 is null")

            // Update rule for each parameter
            W1 = W1.sub(dW1.mul(learningRate))
            b1 = b1.sub(db1.mul(learningRate))
            W2 = W2.sub(dW2.mul(learningRate))
            b2 = b2.sub(db2.mul(learningRate))

            return mutableMapOf(
                "W1" to W1,
                "b1" to b1,
                "W2" to W2,
                "b2" to b2
            )


        }

        /**
         * Arguments:
         *  X : dataset of shape (2, number of examples)
         *  Y : labels of shape (1, number of examples)
         *  n_h : size of the hidden layer
         *  num_iterations : Number of iterations in gradient descent loop
         *  print_cost : if True, print the cost every 1000 iterations
         *
         *  Returns:
         *  parameters : parameters learnt by the model
         */
        fun nn_model(
            X: INDArray,
            Y: INDArray,
            n_h: Long,
            num_iterations: Int = 10000,
            print_cost: Boolean = false
        ): MutableMap<String, INDArray> {

            Nd4j.getRandom().setSeed(3)

            val (n_x, _, n_y) = layerSizes(X, Y)

            // Initialize parameters
            var parameters = initParameters(n_x, n_h, n_y)

            // Loop (gradient descent)​
            for (i in 0..num_iterations) {
                // Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
                val (A2, cache) = forwardPropagation(
                    X,
                    parameters
                )

                // Cost function . Inputs : "A2, Y, parameters". Outputs: "cost".
                val cost = computeCost(A2, Y, parameters)

                // Backpropagation.Inputs: "parameters, cache, X, Y". Outputs: "grads".
                val grads = backwardPropagation(
                    parameters,
                    cache,
                    X,
                    Y
                )

                // Gradient descent parameter update . Inputs : "parameters, grads". Outputs: "parameters".
                parameters =
                    updateParameters(parameters, grads)

                // Print the cost every 1000 iterations
                if (print_cost && i % 1000 == 0) {
                    println("Cost after iteration $i: $cost")
                }
            }
            return parameters
        }


        /**
         *
         * Using the learned parameters, predicts a class for each example in X
         *
         * Arguments:
         * parameters : map containing parameters
         * X : input data of size (n_x, m)
         *
         * Returns
         * predictions : vector of predictions of the model
         */
        fun predict(parameters: Map<String, INDArray>, X: INDArray): INDArray {
            //Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
            val (A2, cache) = forwardPropagation(
                X,
                parameters
            )
            return A2.gt(0.5)
        }
    }

}