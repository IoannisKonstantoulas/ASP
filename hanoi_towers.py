import run_clingo

string ="""\
operator(and;nand;or;nor;xor;xnor).
object(a).
value(circle;triangle;square;yes).
argument(isShape;isFilled).

0 {hasArgument(S,P,V,X,Y): value(V)} 1 :- object(S), argument(P), X=1..3, Y=1..3.

%1st row
hasArgument(a,isShape,circle,1,1).
not hasArgument(a,isShape,triangle,1,1).
hasArgument(a,isShape,triangle,1,2).
hasArgument(a,isShape,square,1,3).
hasArgument(a,isFilled,yes,1,3).

%2nd row
hasArgument(a,isShape,square,2,1).
hasArgument(a,isShape,circle,2,2).
hasArgument(a,isShape,triangle,2,3).
hasArgument(a,isFilled,yes,2,3).

%3rd row
hasArgument(a,isShape,triangle,3,1).
hasArgument(a,isShape,square,3,2).
hasArgument(a,isShape,circle,3,3).
hasArgument(a,isFilled,yes,3,3).



%NAND clause
hasArgument(S,P,V,X,3) :- hasOperator(nand,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(nand,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(nand,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(nand,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nand,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nand,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nand,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nand,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.

%AND clause
not hasArgument(S,P,V,X,3) :- hasOperator(and,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(and,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(and,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(and,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
:- hasOperator(and,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(and,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(and,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(and,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.

%NOR clause
hasArgument(S,P,V,X,3) :- hasOperator(nor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(nor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(nor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(nor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(nor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.

%OR clause
not hasArgument(S,P,V,X,3) :- hasOperator(or,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(or,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(or,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(or,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
:- hasOperator(or,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(or,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(or,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(or,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.

%XNOR clause
hasArgument(S,P,V,X,3) :- hasOperator(xnor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(xnor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(xnor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(xnor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xnor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xnor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xnor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xnor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.

%XOR clause
not hasArgument(S,P,V,X,3) :- hasOperator(xor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(xor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
hasArgument(S,P,V,X,3) :- hasOperator(xor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
not hasArgument(S,P,V,X,3) :- hasOperator(xor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.
:- hasOperator(xor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value(V), X=1..3.

1 {hasOperator(O,S,P): operator(O)} 3 :- object(S), argument(P).

#show hasArgument/5.

"""

run_clingo.run(string)