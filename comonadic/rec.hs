{-# LANGUAGE DeriveFunctor #-}

data Natural =
    One
    | Succ Natural

data List a =
    Empty
    | Cons a (List a)

-- Fixed-point type: Fix
newtype Fix f = Fix (f (Fix f))

data NatF r =
    OneF
    | SuccF r
    deriving (Functor, Show)

type Nat = Fix NatF

one :: Nat
one = Fix OneF

succ :: Nat -> Nat
succ = Fix . SuccF

-- Another way to write Fix
data Program f = Running (f (Program f))

-- So think of Fix as defining a program that runs until f decides
-- to terminate (base case) -- i.e. f is an "instruction set" for the program.
data Instruction r =
    Increment r
    | Decrement r
    | Terminate
    deriving (Functor, Show)

increment :: Program Instruction -> Program Instruction
increment = Running . Increment

decrement :: Program Instruction -> Program Instruction
decrement = Running . Decrement

terminate :: Program Instruction
terminate = Running Terminate

program :: Program Instruction
program =
    increment
    . increment
    . decrement
    $ terminate

-- Free = Fix + parameterized types
data Free f a =
    Free (f (Free f a))
    | Pure a
    deriving Functor

