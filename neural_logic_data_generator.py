import run_clingo

string ="""\

man(a).
woman(b).
father(b,a).

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

not father(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not_father(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not father(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.
not_father(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.
not mother(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not_mother(X,Y) :- father(Y,X), person(X), person(Y), X!=Y.
not mother(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.
not_mother(X,Y) :- mother(Y,X), person(X), person(Y), X!=Y.

#show man/1.
#show woman/1.
#show not_man/1.
#show not_woman/1.
#show father/2.
#show mother/2.
#show not_father/2.
#show not_mother/2.

"""

run_clingo.run(string)