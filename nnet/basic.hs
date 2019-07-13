import Numeric.LinearAlgebra as LA

-- Sigmoid activation function
sigmoid :: Matrix Double -> Matrix Double
sigmoid = cmap (\z -> recip $ 1.0 + exp (-z))

sigmoidGrad x dy = dy * y * (ones - y)
    where y = sigmoid x
          ones = (rows x) >< (cols x) $ repeat 1.0

-- Single-layer forward pass. Returns logits + hidden layer
-- for backpropagation.
forward :: Matrix Double -> Matrix Double -> [Matrix Double]
forward x w =
    let h = x LA.<> w
        y = sigmoid h
    in [h, y]

linearGrad x dy = cmap (/ m) $ tr' x LA.<> dy
    where m = fromIntegral $ rows x

-- MSE loss function
loss :: Matrix Double -> Matrix Double -> Double
loss y target = let diff = y - target
                in sumElements $ cmap (^2) diff

lossGrad :: Matrix Double -> Matrix Double -> Matrix Double
lossGrad y target = let diff = y - target
                    in cmap (*2) diff       

-- Backpropagation of gradients (through a single layer)
backprop :: (Matrix Double, Matrix Double) -> Matrix Double -> Matrix Double
backprop (x, y) w = dLdW
    where
        -- Forward pass
        [h, y_pred] = forward x w
        -- Backwards pass
        dLdy = lossGrad y_pred y
        dLdh = sigmoidGrad h dLdy
        dLdW = linearGrad x dLdh

-- Gradient descent update step
descend gradF iterN lr x0 =
    take iterN (iterate update x0)
    where
        update x = x - lr * (gradF x)

-- Weight initializer
initializeWeights (num_inputs, num_outputs) = do
    -- Xavier initialization
    let k = sqrt (1.0 / fromIntegral num_inputs)
    w <- LA.randn num_inputs num_outputs
    return $ cmap (* k) w

-- Test!
main = do
    -- Load data
    x <- LA.loadMatrix "data/x.dat"
    y <- LA.loadMatrix "data/y.dat"
    -- Build neural network model
    putStrLn "-> Initializing network..."
    let (num_inputs, num_outputs) = (4, 3)
    w <- initializeWeights (num_inputs, num_outputs)
    -- Forward pass
    let [_, y_pred_init] = forward x w
    putStrLn $ "Initial loss: " ++ show (loss y_pred_init y)
    -- Training
    putStrLn "-> Training network..."
    let num_epochs = 5000
    let lr = 0.01
    let w_trained = last $ descend (backprop (x,y)) num_epochs lr w
    print $ w_trained
    -- Evaluate
    let [_, y_pred] = forward x w_trained
    putStrLn $ "Trained loss: " ++ show (loss y_pred y)