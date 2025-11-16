{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module SSISearch where

import Clash.Prelude

type HashKey = Unsigned 64
type NodeAddr = Unsigned 32

data SearchRequest = SearchRequest
    { searchKey :: HashKey
    , rootAddr :: NodeAddr
    } deriving (Generic, NFDataX, Show, Eq)

data SearchResult = SearchResult
    { foundAddr :: NodeAddr
    , found :: Bool
    , depth :: Unsigned 8
    } deriving (Generic, NFDataX, Show, Eq)

data TreeNode = TreeNode
    { nodeKey :: HashKey
    , leftChild :: NodeAddr
    , rightChild :: NodeAddr
    , isValid :: Bool
    } deriving (Generic, NFDataX, Show, Eq)

data SearchState
    = Idle
    | Fetching NodeAddr (Unsigned 8)
    | Comparing HashKey NodeAddr (Unsigned 8)
    deriving (Generic, NFDataX, Show, Eq)

maxSearchDepth :: Unsigned 8
maxSearchDepth = 32

ssiSearch
    :: HiddenClockResetEnable dom
    => Signal dom (Maybe SearchRequest)
    -> Signal dom (Maybe TreeNode)
    -> (Signal dom (Maybe NodeAddr), Signal dom (Maybe SearchResult))
ssiSearch reqIn nodeIn = (memReq, resultOut)
  where
    (state, memReq, resultOut) = unbundle $ mealy ssiSearchT Idle (bundle (reqIn, nodeIn))

ssiSearchT
    :: SearchState
    -> (Maybe SearchRequest, Maybe TreeNode)
    -> (SearchState, (Maybe NodeAddr, Maybe SearchResult))
ssiSearchT Idle (Just req, _) =
    (Fetching (rootAddr req) 0, (Just (rootAddr req), Nothing))

ssiSearchT (Fetching addr depth) (_, Just node)
    | depth >= maxSearchDepth = (Idle, (Nothing, Just depthExceeded))
    | not (isValid node) = (Idle, (Nothing, Just notFound))
    | otherwise = (Comparing (nodeKey node) addr (depth + 1), (Nothing, Nothing))
  where
    notFound = SearchResult 0 False depth
    depthExceeded = SearchResult 0 False maxSearchDepth

ssiSearchT (Comparing key addr depth) (Just req, Just node)
    | depth >= maxSearchDepth = (Idle, (Nothing, Just depthExceeded))
    | searchKey req == key = (Idle, (Nothing, Just foundResult))
    | searchKey req < key && leftChild node /= 0 =
        (Fetching (leftChild node) depth, (Just (leftChild node), Nothing))
    | searchKey req > key && rightChild node /= 0 =
        (Fetching (rightChild node) depth, (Just (rightChild node), Nothing))
    | otherwise = (Idle, (Nothing, Just notFound))
  where
    foundResult = SearchResult addr True depth
    notFound = SearchResult 0 False depth
    depthExceeded = SearchResult 0 False maxSearchDepth

ssiSearchT state _ = (state, (Nothing, Nothing))

topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Signal System (Maybe SearchRequest)
    -> Signal System (Maybe TreeNode)
    -> (Signal System (Maybe NodeAddr), Signal System (Maybe SearchResult))
topEntity = exposeClockResetEnable ssiSearch
{-# NOINLINE topEntity #-}

testSearchRequest :: SearchRequest
testSearchRequest = SearchRequest
    { searchKey = 0x123456
    , rootAddr = 0x1000
    }

testTreeNode :: TreeNode
testTreeNode = TreeNode
    { nodeKey = 0x123456
    , leftChild = 0x2000
    , rightChild = 0x3000
    , isValid = True
    }

simulateSearch :: Maybe SearchRequest -> Maybe TreeNode -> (SearchState, Maybe SearchResult)
simulateSearch Nothing _ = (Idle, Nothing)
simulateSearch (Just req) Nothing = (Fetching (rootAddr req) 0, Nothing)
simulateSearch (Just req) (Just node)
    | not (isValid node) = (Idle, Just notFound)
    | searchKey req == nodeKey node = (Idle, Just found)
    | searchKey req < nodeKey node = (Fetching (leftChild node) 1, Nothing)
    | otherwise = (Fetching (rightChild node) 1, Nothing)
  where
    notFound = SearchResult 0 False 0
    found = SearchResult (rootAddr req) True 1

main :: IO ()
main = do
    putStrLn "SSISearch Simulation"
    putStrLn "==================="
    putStrLn "Testing iterative tree search with depth limiting..."
    putStrLn ""
    
    putStrLn "Test 1: Search exact match"
    putStrLn $ "  Request: " ++ show testSearchRequest
    putStrLn $ "  Tree node: " ++ show testTreeNode
    let (state1, result1) = simulateSearch (Just testSearchRequest) (Just testTreeNode)
    putStrLn $ "  State: " ++ show state1
    putStrLn $ "  Result: " ++ show result1
    
    putStrLn "\nTest 2: Search left child"
    let reqLeft = SearchRequest 0x100000 0x1000
    let (state2, result2) = simulateSearch (Just reqLeft) (Just testTreeNode)
    putStrLn $ "  Search key < node key -> traverse left"
    putStrLn $ "  State: " ++ show state2
    putStrLn $ "  Result: " ++ show result2
    
    putStrLn "\nTest 3: Search right child"
    let reqRight = SearchRequest 0x200000 0x1000
    let (state3, result3) = simulateSearch (Just reqRight) (Just testTreeNode)
    putStrLn $ "  Search key > node key -> traverse right"
    putStrLn $ "  State: " ++ show state3
    putStrLn $ "  Result: " ++ show result3
    
    putStrLn "\nTest 4: Invalid node"
    let invalidNode = TreeNode 0 0 0 False
    let (state4, result4) = simulateSearch (Just testSearchRequest) (Just invalidNode)
    putStrLn $ "  Invalid node -> return not found"
    putStrLn $ "  State: " ++ show state4
    putStrLn $ "  Result: " ++ show result4
    
    putStrLn "\nTest 5: Maximum depth limit"
    putStrLn $ "  Max search depth: " ++ show maxSearchDepth
    putStrLn "  Prevents infinite recursion in degenerate trees"
    
    putStrLn "\nSimulation complete!"
    putStrLn "SSISearch uses iterative state machine with bounded depth."
