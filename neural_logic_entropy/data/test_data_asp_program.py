import run_clingo

asp_program =  """\
person(a).
person(b).

not man(X) :- woman(X), person(X).
not_man(X) :- woman(X), person(X).
not woman(X) :- man(X), person(X).
not_woman(X) :- man(X), person(X).
man(X) :- not_woman(X), person(X).
woman(X) :- not_man(X), person(X).

not father(Y,X) :- not_man(X), person(X), person(Y), X!=Y.
not_father(Y,X) :- not_man(X), person(X), person(Y), X!=Y.
not mother(Y,X) :- not_woman(X), person(X), person(Y), X!=Y.
not_mother(Y,X) :- not_woman(X), person(X), person(Y), X!=Y.

man(X) :- father(Y,X), person(X), person(Y).
woman(X) :- mother(Y,X), person(X), person(Y).

not_parent(Y,X) :- not_father(Y,X), not_mother(Y,X), person(X), person(Y).

not not_parent(Y,X) :- parent(Y,X), person(X), person(Y).
not parent(Y,X) :- not_parent(Y,X), person(X), person(Y).

parent(Y,X) :- father(Y,X), person(X), person(Y).
parent(Y,X) :- mother(Y,X), person(X), person(Y).

not father(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not_father(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not father(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.
not_father(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.
not mother(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not_mother(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not mother(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.
not_mother(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.

not parent(X,Y) :- parent(Y,X), person(X), person(Y), X!=Y.
not_parent(X,Y) :- parent(Y,X), person(X), person(Y), X!=Y.

father(Y,X) :- parent(Y,X), man(X), person(X), person(Y), X!=Y.
mother(Y,X) :- parent(Y,X), woman(X), person(X), person(Y), X!=Y.

#show man/1.
#show woman/1.
#show not_man/1.
#show not_woman/1.
#show parent/2.
#show not_parent/2.
#show father/2.
#show mother/2.
#show not_father/2.
#show not_mother/2.

"""

test_string = """\
man(a).
woman(1).
parent(a,b).

"""
# run_clingo.run(test_string + asp_program)
