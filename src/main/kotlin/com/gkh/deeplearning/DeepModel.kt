package com.gkh.deeplearning

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.lang.IllegalArgumentException
import kotlin.math.floor

typealias nj = Nd4j
typealias Matrix = INDArray
typealias LinearForwardCache = Triple<Matrix, Matrix, Matrix>
typealias LinearForwardTuple = Pair<Matrix, LinearForwardCache>
typealias LinearForwardAndActivationCache = Pair<LinearForwardCache, Matrix>
typealias LinearActivationForwardTuple = Pair<Matrix, LinearForwardAndActivationCache>
typealias LinearModelForwardTuple = Pair<Matrix, List<LinearForwardAndActivationCache>>
typealias LinearBackwardTriple = Triple<Matrix, Matrix, Matrix>
typealias LinearActivationBackwardTriple = Triple<Matrix, Matrix, Matrix>

/**
 * L-Layer implementation of Deep Learning algorithm, based on lessons from
 * Andrew Ng's deep learning course in Coursera
 */
class DeepModel {

    companion object {

        /**
         * Helper function for relu
         */
        fun relu(Z: Matrix): Pair<Matrix, Matrix> {
            val A = Transforms.relu(Z)
            return Pair(A, Z)
        }

        /**
         * Helper function for sigmoid
         */
        fun sigmoid(Z: Matrix): Pair<Matrix, Matrix> {
            val A = Transforms.sigmoid(Z)
            return Pair(A, Z)
        }

        /**
         * Helper function for relu derivative
         */
        fun reluBackward(dA: Matrix, activationCache: Matrix): Matrix {
            return dA.mul(Transforms.relu6(activationCache, true))
        }

        /**
         * Helper function for sigmoid backward
         */
        fun sigmoidBackward(dA: Matrix, activationCache: Matrix): Matrix {
            return dA.mul(Transforms.sigmoidDerivative(activationCache, true))
        }

        /**
         * Arguments:
         *  layer_dims -- array (list) containing the dimensions of each layer in our network
         * Returns:
         *  parameters -- dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
         *  Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
         *  bl -- bias vector of shape (layer_dims[l], 1)
         */
        fun initParameters(layerDims: List<Long>): MutableMap<String, Matrix> {
            val parameters = mutableMapOf<String, Matrix>()
            val L = layerDims.size
            for (l in 1 until L) {
                parameters["W$l"] = nj.randn(DataType.DOUBLE, layerDims[l], layerDims[l - 1]).mul(0.01)
                parameters["b$l"] = nj.zeros(DataType.DOUBLE, layerDims[l], 1)

                assert(parameters["W$l"]!!.shape().toList() == listOf(layerDims[l], layerDims[l - 1]))
                assert(parameters["b$l"]!!.shape().toList() == listOf(layerDims[l], 1))
            }
            return parameters
        }

        /**
         *
         * Implement the linear part of a layer's forward propagation.
         *
         * Arguments:
         *   A -- activations from previous layer (or input data): (size of previous layer, number of examples)
         *   W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
         *   b -- bias vector, numpy array of shape (size of the current layer, 1)
         *
         * Returns:
         *   Z -- the input of the activation function, also called pre-activation parameter
         *   cache -- a tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
         */
        fun linearForward(A: Matrix, W: Matrix, b: Matrix): LinearForwardTuple {
            val Z = (W.mmul(A)).add(b)
            assert(Z.shape().toList() == listOf(W.shape()[0], A.shape()[1]))

            val cache = Triple(A, W, b)
            return Pair(Z, cache)
        }

        /**
         *  Implement the forward propagation for the LINEAR->ACTIVATION layer
         *  Arguments:
         *   A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
         *   W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
         *   b -- bias vector, numpy array of shape (size of the current layer, 1)
         *   activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
         *
         *  Returns:
         *   A -- the output of the activation function, also called the post-activation value
         *   cache -- a  tuple containing "linear_cache" and "activation_cache";
         *   stored for computing the backward pass efficientl
         */
        fun linearActivationForward(
            A_prev: Matrix,
            W: Matrix,
            b: Matrix,
            activation: String
        ): LinearActivationForwardTuple {
            if (activation == "sigmoid") {
                val (Z, linearCache) = linearForward(A_prev, W, b)
                val (A, activationCache) = sigmoid(Z)
                assert(A.shape().toList() == listOf(W.shape()[0], A_prev.shape()[1]))
                val cache = Pair(linearCache, activationCache)
                return Pair(A, cache)
            } else if (activation == "relu") {
                val (Z, linearCache) = linearForward(A_prev, W, b)
                val (A, activationCache) = relu(Z)
                assert(A.shape().toList() == listOf(W.shape()[0], A_prev.shape()[1]))
                val cache = Pair(linearCache, activationCache)
                return Pair(A, cache)
            }
            throw IllegalArgumentException("Unsupported activation: $activation")
        }

        /**
         * Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
         * Arguments:
         *  X -- data, numpy array of shape (input size, number of examples)
         *  parameters -- output of initialize_parameters_deep()
         * Returns:
         *  AL -- last post-activation value
         *  caches -- list of caches containing:
         *  every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
         */
        fun LModelForward(
            X: Matrix,
            parameters: MutableMap<String, Matrix>
        ): LinearModelForwardTuple {
            val caches = mutableListOf<LinearForwardAndActivationCache>()
            var A = X
            val L = floor(parameters.size / 2.0).toInt() // number of layers in the neural network

            //Implement[LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
            for (l in 1 until L) {
                val A_prev = A
                val (A_, cache) = linearActivationForward(
                    A_prev,
                    parameters["W$l"]!!,
                    parameters["b$l"]!!,
                    activation = "relu"
                )
                A = A_
                caches.add(cache)
            }

            // Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
            val (AL, cache) = linearActivationForward(
                A,
                parameters["W$L"]!!,
                parameters["b$L"]!!,
                activation = "sigmoid"
            )
            caches.add(cache)
            assert(AL.shape().toList() == listOf(1, X.shape()[1]))
            return Pair(AL, caches)
        }

        /**
         * Implement the cost function defined by equation (7).
         * Arguments:
         *   AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
         *   Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
         * Returns:
         *  cost -- cross-entropy cost
         */
        fun computeCost(AL: Matrix, Y: Matrix): Double {
            // number of example
            val m = Y.shape()[1]
            // Compute the cross-entropy cost
            val a = (Transforms.log(AL)).mul(Y)
            val b = (Y.rsub(1)).mul(Transforms.log(AL.rsub(1)))
            val logprobs = a.add(b)
            val cost = (logprobs.sum()).mul(-1.0 / m)

            // makes sure cost is the dimension we expect.
            val result = cost.getDouble(0)

            assert(result is Double)

            return result
        }

        /**
         * Implement the linear portion of backward propagation for a single layer (layer l)
         *​
         * Arguments:
         *  dZ -- Gradient of the cost with respect to the linear output (of current layer l)
         *  cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
         *​
         * Returns:
         *  dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
         *  dW -- Gradient of the cost with respect to W (current layer l), same shape as W
         *  db -- Gradient of the cost with respect to b (current layer l), same shape as b
         */
        fun linearBackward(dZ: Matrix, cache: LinearForwardCache): LinearBackwardTriple {
            val (A_prev, W, b) = cache
            val m = A_prev.shape()[1]

            val dW = (dZ.mmul(A_prev.transpose())).mul(1.0 / m)
            val db = (dZ.sum(true, 1)).mul(1.0 / m)
            val dA_prev = (W.transpose().mmul(dZ))

            assert(dA_prev.shape().contentEquals(A_prev.shape()))
            assert(dW.shape().contentEquals(W.shape()))
            assert(db.shape().contentEquals(b.shape()))

            return Triple(dA_prev, dW, db)
        }

        /**
         * Implement the backward propagation for the LINEAR->ACTIVATION layer.
         *
         * Arguments:
         *  dA -- post-activation gradient for current layer l
         *  cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
         *  activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
         *
         * Returns:
         *  dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
         *  dW -- Gradient of the cost with respect to W (current layer l), same shape as W
         *  db -- Gradient of the cost with respect to b (current layer l), same shape as b
         */
        fun linearActivationBackward(
            dA: Matrix,
            cache: LinearForwardAndActivationCache,
            activation: String
        ): LinearActivationBackwardTriple {
            val (linearCache, activationCache) = cache
            if (activation == "relu") {
                val dZ = reluBackward(dA, activationCache)
                val (dA_prev, dW, db) = linearBackward(dZ, linearCache)
                return Triple(dA_prev, dW, db)
            } else if (activation == "sigmoid") {
                val dZ = sigmoidBackward(dA, activationCache)
                val (dA_prev, dW, db) = linearBackward(dZ, linearCache)
                return Triple(dA_prev, dW, db)
            }
            throw IllegalArgumentException("Unsupported activation: $activation")
        }

        /**
         * Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
         *
         * Arguments:
         *  AL -- probability vector, output of the forward propagation (L_model_forward())
         *  Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
         *  caches -- list of caches containing:
         *  every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
         *  the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
         *
         * Returns:
         *  grads -- A dictionary with the gradients
         *  grads["dA" + str(l)] = ...
         *  grads["dW" + str(l)] = ...
         *  grads["db" + str(l)] = ...
         */
        fun LModelBackward(
            AL: Matrix,
            Y: Matrix,
            caches: List<LinearForwardAndActivationCache>
        ): MutableMap<String, Matrix> {
            val grads = mutableMapOf<String, Matrix>()
            val L = caches.size // the number of layers
            val m = AL.shape()[1]
            val Y = Y.reshape(*AL.shape()) // after this line, Y is the same shape as AL

            // Initializing the backpropagation
            val dAL = ((Y.div(AL)).sub((Y.rsub(1)).div(AL.rsub(1)))).neg()

            // Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
            var currentCache = caches[L - 1]
            val (dA_prev, dWL, dbL) = linearActivationBackward(dAL, currentCache, activation = "sigmoid")
            val L_prev = (L - 1).toString()
            grads["dA$L_prev"] = dA_prev
            grads["dW$L"] = dWL
            grads["db$L"] = dbL

            // Loop from l = L - 2 to 0
            for (l in (L - 2) downTo 0) {
                // lth layer :(RELU -> LINEAR) gradients.
                // Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
                currentCache = caches[l]
                val L_next = (l + 1).toString()
                val (dA_prev_temp, dW_temp, db_temp) = linearActivationBackward(
                    grads["dA$L_next"]!!,
                    currentCache,
                    activation = "relu"
                )
                grads["dA$l"] = dA_prev_temp
                grads["dW$L_next"] = dW_temp
                grads["db$L_next"] = db_temp
            }
            return grads
        }

        /**
         * Update parameters using gradient descent
         *
         * Arguments:
         *  parameters --  dictionary containing your parameters
         *  grads --  dictionary containing your gradients, output of L_model_backward
         *
         * Returns:
         *  parameters --  dictionary containing your updated parameters
         *          parameters["W" + str(l)] = ...
         *          parameters["b" + str(l)] = ...
         */
        fun updateParameters(
            parametersIn: Map<String, Matrix>,
            grads: Map<String, Matrix>, learning_rate: Double
        ): MutableMap<String, Matrix> {
            val parameters = parametersIn.toMutableMap()
            val L = floor(parameters.size / 2.0).toInt() // number of layers in the neural network

            // Update rule for each parameter . Use a for loop.
            for (l in 0 until L) {
                val l_next = (l + 1)
                parameters["W$l_next"] = parameters["W$l_next"]!!.sub(grads["dW$l_next"]!!.mul(learning_rate))
                parameters["b$l_next"] = parameters["b$l_next"]!!.sub(grads["db$l_next"]!!.mul(learning_rate))
            }

            return parameters
        }

        /**
         * Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
         *
         * Arguments:
         *  X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
         *  Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
         * layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
         * learning_rate -- learning rate of the gradient descent update rule
         * num_iterations -- number of iterations of the optimization loop
         * print_cost -- if True, it prints the cost every 100 steps
         *
         * Returns:
         *  parameters -- parameters learnt by the model. They can then be used to predict.
         */
        fun train(
            X: Matrix,
            Y: Matrix,
            layersDims: List<Long>,
            learningRate: Double = 0.0075,
            numIterations: Int = 3000,
            printCost: Boolean = false
        ): MutableMap<String, Matrix> {

            //nj.getRandom().setSeed(3)

            val costs = mutableListOf<Double>() //keep track of cost

            // Parameters initialization. (≈ 1 line of code)
            var parameters = initParameters(layersDims)

            // Loop (gradient descent)
            for (i in 0 until numIterations) {
                // Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
                val (AL, caches) = LModelForward(X, parameters)

                // Compute cost.
                val cost = computeCost(AL, Y)

                // Backward propagation.
                val grads = LModelBackward(AL, Y, caches)

                // Update parameters.
                parameters = updateParameters(parameters, grads, learningRate)

                // Print the cost every 100 training example
                if (printCost && i % 100 == 0) {
                    println("Cost after iteration $i: $cost")
                }

                if (printCost && i % 100 == 0) {
                    costs.add(cost)
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
        fun predict(parameters: MutableMap<String, INDArray>, X: INDArray): INDArray {
            //Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
            val (A, cache) = LModelForward(X, parameters)
            return A.gt(0.5)
        }

    }
}