%% Facts listing

value("zero", 0).
value("one", 1).       value("two", 2).        value("three", 3).
value("four", 4).      value("five", 5).       value("six", 6).
value("seven", 7).     value("eight", 8).      value("nine", 9).
value("ten", 10).      value("eleven", 11).    value("twelve", 12).
value("thirteen", 13). value("fourteen", 14).  value("fifteen", 15).
value("sixteen", 16).  value("seventeen", 17). value("eighteen", 18).
value("nineteen", 19).

value("twenty", 20).   value("thirty", 30).   value("forty", 40).
value("fifty", 50).    value("sixty", 60).    value("seventy", 70).
value("eighty", 80).   value("ninety", 90).	value("hundred", 100).
value("one-thousand", 1000).


%% calculation rules definition

% if there are no more words then we can put the accumulator to the result because it will be added 
% to the total (as the last word of the sequence).
compute([], Acc, Acc).

% We ignore any "and" words like in "one hundred and five"
compute(["and"|Rem], Acc, Total) :- !, compute(Rem, Acc, Total). % we use the ! to stop backtracking

% if the word is "hundred", we multiply the accumulator by 100
compute(["hundred"|Rem], Acc, Total) :- !, NouvelAcc is Acc * 100, compute(Rem, NouvelAcc, Total).

% If no special case, we find the value of the word with value(X, Y) fact and add it's value to the accumulator
compute([Mot|Rem], Acc, Total) :- value(Mot, Val), !, NouvelAcc is Acc + Val, compute(Rem, NouvelAcc, Total).



%% In the "main" predicate, we clean the string then launch the computation predicate.

to_num(String) :-
    string_lower(String, LowerString), % prevent any upper cases fault
    % We split the string using spaces and dashes "one-hundred and thirty-six" --> [one, hundred, and, thirty, six]
    split_string(LowerString, " -", "", WordList),
    compute(WordList, 0, Res), % compute the number from the cleaned word list
    write(Res).

%% Websites visited:
% https://www.swi-prolog.org/pldoc/man?predicate=split_string/4
% https://www.swi-prolog.org/pldoc/man?predicate=string_lower/2
% https://staff.fnwi.uva.nl/u.endriss/teaching/prolog/prolog.pdf