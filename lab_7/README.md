# Lab 7 - Prolog Number Converter

## Variant 4

The objective of this exercise is to convert English words of a written number into numerical digits.

The supported range is from 0 to 1000.

Example:

```prolog
?- to_num("ninety nine").
99

?- to_num("one hundred and one").
101
```

## Files

- `program.pl`: Prolog source code.
- `README.md`: Instructions to run the code.
- `Report EARIN lab6.pdf`: Short report explaining the solution.

## How to run the code

You can run the program using SWI-Prolog.

First, open SWI-Prolog in the folder containing the file `lab7_cg104_g19_v4_Morra-Fischer_Capomaggio.pl`.


After that, you can query the predicate `to_num/1`.

Example:

```prolog
?- to_num("ninety nine").
99
true.
```

You can also use the predicate `to_num/2` to store the result in a variable:

```prolog
?- to_num("ninety nine", N).
N = 99.
```

## Example queries

```prolog
?- to_num("zero").
0

?- to_num("twenty five").
25

?- to_num("twenty-five").
25

?- to_num("one hundred").
100

?- to_num("one hundred and one").
101

?- to_num("seven hundred thirty six").
736

?- to_num("one thousand").
1000
```

## Invalid examples

Some incorrect number expressions are rejected:

```prolog
?- to_num("six five", N).
false.

?- to_num("ten ten", N).
false.

?- to_num("twenty ten", N).
false.

?- to_num("hundred five", N).
false.

?- to_num("nine hundred hundred", N).
false.
```

## Notes

The program accepts both spaces and hyphens between words.

For example:

```prolog
?- to_num("twenty-five", N).
N = 25.
```

The word `"and"` is ignored, so both forms are accepted:

```prolog
?- to_num("one hundred and five", N).
N = 105.

?- to_num("one hundred five", N).
N = 105.
```
