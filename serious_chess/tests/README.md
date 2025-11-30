# DESCRIPTION:

One of the basic chess endgames is where only three pieces remain on the board: the two kings and one rook. By 1970 it was proved that the player having the rook can win the game in at most 16 moves. Can you write code that plays this endgame that well?

## Short Overview

Your code will play as white and the testing code will play as black.

Each test case provides you with an initial game setup, consisting of the positions of three pieces: the white king, the white rook, and the black king. Then you will get a valid move for the black king, to which you must reply with a move with a white piece. You will then get another black move for which you should reply with the next white move, ...etc. The testing code will stop this exchange of moves once the game ends, the rook is lost, or an invalid move is made.

Your code must be able to give a checkmate within 16 moves.

At the end of this description you'll find the relevant rules of chess.

## Details of the Task

Write a class WhitePlayer with:

A constructor taking a string as argument, listing the positions of the three pieces in the format Kxy,Rxy - Kxy where xy are coordinates in algebraic notation and the other characters are literal.
Examples of such strings are Kf8,Rd5 - Ke3 and Kb6,Rc5 - Kb3.

The coordinates will always appear in this order and format, and will always be valid. They define the positions of the white king, the white rook, and the black king in that order.

In all provided positions it is black's turn to play a move.

A method play, which takes a string as argument and returns a string.
The testing code will call this method to pass as argument a legal black king move. Examples are Kf3, Kb2, ... always starting with K followed by the new position of the black king.

The method should return the move that white will make in response to this black move. It should have the same format. For example: Rd7, or Kb5. The method should update the state of the game, reflecting that both the given black move and the produced white move have been played. The next call of this method will provide a black move applicable to that modified game state.

The testing code will keep calling the method with a next black move until one of the following happens (assuming no exceptions are raised):

the method returns a value that does not represent a valid move (e.g. wrong data type, wrong format, or violating chess rules): the test fails.
black can capture the rook: the test fails. The capturing move is mentioned in the failure message with an additional "x", like Kxb4.
black cannot make a valid move. There are three possibilities:
Checkmate after at most 16 white moves: the test succeeds.
Checkmate after more than 16 white moves: the test fails. Note that the test is not interrupted after 16 white moves so at least you can see how many moves it took.
Stalemate: the test fails.
The threefold position or 50-move rule applies: the test fails.

##Tests

The example test cases can be finished with one move. If somehow your code takes up to 16 moves for a mate, the tests will pass. But it seems reasonable that your code should detect mate-in-one positions.

The full tests include around 40 fixed tests with differing levels of difficulty.

There are 500 random tests.

