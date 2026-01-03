import run_clingo

asp_program =  """\
person(a).
person(b).

:- man(X), not_man(X).
:- woman(X), not_woman(X).
:- parent(X,Y), not_parent(X,Y).
:- father(X,Y), not_father(X,Y).
:- mother(X,Y), not_mother(X,Y).

:- woman(X), man(X), person(X).
:- not_woman(X), not_man(X), person(X).

not_man(X) :- woman(X), person(X).
not_woman(X) :- man(X), person(X).
man(X) :- not_woman(X), person(X).
woman(X) :- not_man(X), person(X).

:- father(X,Y), not_man(X), person(X), person(Y), X!=Y.
not_father(X,Y) :- not_man(X), person(X), person(Y), X!=Y.
:- mother(X,Y), not_woman(X), person(X), person(Y), X!=Y.
not_mother(X,Y) :- not_woman(X), person(X), person(Y), X!=Y.

man(X) :- father(X,Y), person(X), person(Y).
woman(X) :- mother(X,Y), person(X), person(Y).

:- parent(X,Y), not_father(X,Y), not_mother(X,Y), person(X), person(Y).
not_parent(X,Y) :- not_father(X,Y), not_mother(X,Y), person(X), person(Y).

not_mother(X,Y) :- not_parent(X,Y),  person(X), person(Y).
not_father(X,Y) :- not_parent(X,Y),  person(X), person(Y).

:- parent(X,Y), not_parent(X,Y), person(X), person(Y).

parent(X,Y) :- father(X,Y), person(X), person(Y).
parent(X,Y) :- mother(X,Y), person(X), person(Y).

:- father(X,Y), parent(Y,X), person(X), person(Y), X!=Y.
not_father(X,Y) :- parent(Y,X), person(X), person(Y), X!=Y.
:- mother(X,Y), parent(Y,X), person(X), person(Y), X!=Y.
not_mother(X,Y) :- parent(Y,X), person(X), person(Y), X!=Y.
:- parent(X,Y), parent(Y,X), person(X), person(Y), X!=Y.
not_parent(X,Y) :- parent(Y,X), person(X), person(Y), X!=Y.

father(X,Y) :- parent(X,Y), man(X), person(X), person(Y), X!=Y.
mother(X,Y) :- parent(X,Y), woman(X), person(X), person(Y), X!=Y.

father(X,Y) :- parent(X,Y), not_mother(X,Y), person(X), person(Y), X!=Y.
mother(X,Y) :- parent(X,Y), not_father(X,Y), person(X), person(Y), X!=Y.

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
woman(b).
not_father(a,b).
parent(a,b).

"""
# run_clingo.run(test_string + asp_program, log_level="DEBUG")
