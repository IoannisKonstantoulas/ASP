import run_clingo



string = """\
#const m = 3.
#const n = 3.
box(2,2,1).
box(3,1,2).
box(3,1,3).

{ rotate_box(1..2,A,B,ID) } ==1:- box(A,B,ID).
{ placed_box(1..m+1-A,1..n+1-B,A,B,ID) } == 1 :- rotate_box(1,A,B,ID).
{ placed_box(1..m+1-A,1..n+1-B,A,B,ID) } == 1 :- rotate_box(2,B,A,ID).

{ placed_box(X..A-1,Y..B-1,1..m,1..n,1..ID-1; X..A-1,Y..B-1,1..m,1..n,ID+1..6) } == 0:- placed_box(X,Y,A,B,ID).
"""

run_clingo.run(string)