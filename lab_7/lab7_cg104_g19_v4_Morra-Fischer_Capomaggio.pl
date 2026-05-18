%% Facts listing
%% Variant 4: convert English words of a written number into numerical digits.
%% Supported range: 0 to 1000.

value("zero", 0).

unit("one", 1).       unit("two", 2).        unit("three", 3).
unit("four", 4).      unit("five", 5).       unit("six", 6).
unit("seven", 7).     unit("eight", 8).      unit("nine", 9).

teen("ten", 10).      teen("eleven", 11).    teen("twelve", 12).
teen("thirteen", 13). teen("fourteen", 14).  teen("fifteen", 15).
teen("sixteen", 16).  teen("seventeen", 17). teen("eighteen", 18).
teen("nineteen", 19).

tens("twenty", 20).   tens("thirty", 30).   tens("forty", 40).
tens("fifty", 50).    tens("sixty", 60).    tens("seventy", 70).
tens("eighty", 80).   tens("ninety", 90).


%% Utility predicate
%% Remove all occurrences of "and" from the word list.
%% Example: ["one", "hundred", "and", "five"] -> ["one", "hundred", "five"]

remove_and([], []).
remove_and(["and" | Rem], Cleaned) :-
    !,
    remove_and(Rem, Cleaned).
remove_and([Word | Rem], [Word | Cleaned]) :-
    remove_and(Rem, Cleaned).


%% Parsing rules
%% These rules avoid incorrect additions such as "six five", "ten ten", or "twenty ten".

% zero
parse_number(["zero"], 0).

% one thousand
parse_number(["one", "thousand"], 1000).

% numbers from 1 to 99
parse_number(Words, N) :-
    below_hundred(Words, N).

% exact hundreds, for example: "three hundred"
parse_number([UnitWord, "hundred"], N) :-
    unit(UnitWord, U),
    N is U * 100.

% hundreds with a remainder, for example: "three hundred twenty five"
parse_number([UnitWord, "hundred" | Rem], N) :-
    unit(UnitWord, U),
    below_hundred(Rem, R),
    N is U * 100 + R.


%% Numbers below 100

% one, two, ..., nine
below_hundred([Word], N) :-
    unit(Word, N).

% ten, eleven, ..., nineteen
below_hundred([Word], N) :-
    teen(Word, N).

% twenty, thirty, ..., ninety
below_hundred([Word], N) :-
    tens(Word, N).

% twenty one, thirty six, ..., ninety nine
below_hundred([TensWord, UnitWord], N) :-
    tens(TensWord, T),
    unit(UnitWord, U),
    N is T + U.


%% Main predicate with output variable.
%% Example: ?- to_num("ninety nine", N).
%% N = 99.

to_num(String, Res) :-
    string_lower(String, LowerString),
    % Split the string using spaces and hyphens.
    % Example: "one-hundred and thirty-six" -> ["one", "hundred", "and", "thirty", "six"]
    split_string(LowerString, " -", "", WordList),
    remove_and(WordList, CleanWordList),
    parse_number(CleanWordList, Res),
    Res =< 1000.


%% Main predicate matching the exercise examples.
%% Example: ?- to_num("ninety nine").
%% 99

to_num(String) :-
    to_num(String, Res),
    write(Res).


%% Example valid queries:
%% ?- to_num("ninety nine").
%% ?- to_num("one hundred and one").
%% ?- to_num("seven hundred thirty six", N).
%% ?- to_num("twenty-five", N).
%% ?- to_num("one thousand", N).
%%
%% Example invalid queries:
%% ?- to_num("six five", N).          false.
%% ?- to_num("ten ten", N).           false.
%% ?- to_num("twenty ten", N).        false.
%% ?- to_num("hundred five", N).      false.
%% ?- to_num("nine hundred hundred", N). false.

%% Websites visited:
% https://www.swi-prolog.org/pldoc/man?predicate=split_string/4
% https://www.swi-prolog.org/pldoc/man?predicate=string_lower/2
% https://staff.fnwi.uva.nl/u.endriss/teaching/prolog/prolog.pdf