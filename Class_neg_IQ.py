import run_clingo

string ="""\

%FROM HERE BEGIN AUTO-GENERATED INPUTS BY PARSING ML-CLASSIFICATION OUTPUTS THROUGH A pyCode
object(a).
argument(isShape;isFilled).
value((isShape,circle);(isShape,triangle);(isShape,square);(isFilled,yes);(isFilled,no)).

%1st row
hasArgument(a,isShape,circle,1,1).
hasArgument(a,isFilled,no,1,1).
hasArgument(a,isShape,triangle,1,2).
hasArgument(a,isFilled,no,1,2).
hasArgument(a,isShape,square,1,3).
hasArgument(a,isFilled,yes,1,3).

%2nd row
hasArgument(a,isShape,square,2,1).
hasArgument(a,isFilled,no,2,1).
hasArgument(a,isShape,circle,2,2).
hasArgument(a,isFilled,no,2,2).
hasArgument(a,isShape,triangle,2,3).
hasArgument(a,isFilled,yes,2,3).

%3rd row
hasArgument(a,isShape,triangle,3,1).
hasArgument(a,isFilled,no,3,1).
hasArgument(a,isShape,square,3,2).
hasArgument(a,isFilled,no,3,2).
hasArgument(a,isShape,circle,3,3).
hasArgument(a,isFilled,yes,3,3).


%FROM HERE BEGINS THE ASP PROGRAM
operator(and;nand;or;nor;xor;xnor).

1{hasOperator(O,S,P):operator(O)} :- object(S), argument(P).

{hasArgument(S,P,V,X,Y)} :- object(S), argument(P), value((P,V)), X=1..3, Y=1..3.
not hasArgument(S,P,V2,X,Y):- hasArgument(S,P,V,X,Y), object(S), argument(P), value((P,V)), value((P,V2)), V!=V2, X=1..3, Y=1..3.

%NAND clause
:- hasOperator(nand,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(nand,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(nand,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(nand,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.

%AND clause
:- hasOperator(and,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(and,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(and,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(and,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.

%NOR clause
:- hasOperator(nor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(nor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(nor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(nor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.

%OR clause
:- hasOperator(or,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(or,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(or,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(or,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.

%XNOR clause
:- hasOperator(xnor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(xnor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(xnor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(xnor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.

%XOR clause
:- hasOperator(xor,S,P), not hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(xor,S,P), not hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(xor,S,P), hasArgument(S,P,V,X,1), not hasArgument(S,P,V,X,2), not hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.
:- hasOperator(xor,S,P), hasArgument(S,P,V,X,1), hasArgument(S,P,V,X,2), hasArgument(S,P,V,X,3), object(S), argument(P), value((P,V)), X=1..3.

#show hasOperator/3.

"""

run_clingo.run(string)