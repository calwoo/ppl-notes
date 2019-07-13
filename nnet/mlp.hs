import Numeric.LinearAlgebra as LA

-- neural network basics
-- layers: weights, bias, activation
data NNLayer a = Layer (Matrix a) (Matrix a) Activation
data Activation = ReLU
                | Sigmoid
                | Tanh
                | Id

-- gradients: dLdW, dLdb
data Gradients a = Gradients (Matrix a) (Matrix a)
-- a neural net is a stack of layers
type NeuralNetwork = [NNLayer Double]

-- forward propagation



-- backpropagation update step
update :: Double                         -- learning rate
       -> Integer                        -- number of iterations
       -> NeuralNetwork                  -- initial network
       -> (Matrix Double, Matrix Double) -- dataset
update lr iterN net0 data = last $ take iterN (iterate step net0)
    where
        step net = zipWith descend net grads
            where
                (_, grads) = forward net data
        descend :: NNLayer Double
                -> Gradients Double
                -> NNLayer Double
        descend (Layer w b act) (Gradients dW db) =
            Layer (w - lr `scale` dW) (b - lr `scale` db) act
