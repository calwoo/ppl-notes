{-# LANGUAGE LambdaCase #-}

import Control.Monad
import Control.Monad.Free
import qualified System.Random.MWC.Probability as MWC

data DistF r =
    BernoulliF Double (Bool -> r)
    | BetaF Double Double (Double -> r)
    | NormalF Double Double (Double -> r)
    
instance Functor DistF where
    fmap f (BernoulliF p g)   = BernoulliF p (f . g)
    fmap f (BetaF a b g)      = BetaF a b (f . g)
    fmap f (NormalF mu sig g) = NormalF mu sig (f . g)

type Dist = Free DistF