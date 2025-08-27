# bucklin-range
Voting scripts based on the Bucklin-Range method, which is in turn based on 
PTASes of the k-median and uncapacitated facility location problems[1]. 

The Double Bucklin variant is not analogous to the UFLP; instead, that variant
was created by altering the Range variant. 

If I recall correctly, if the corresponding theoretical problem for rated 
ballots is k-median, then the problem corresponding to graded ballots is 
k-center. Unfortunately, the k-center PTASes I've investigated seem to be of 
little use, as they require candidate-candidate distances and not just 
candidate-voter distances.

## References
- [First EM list post, describing Bucklin-Range](http://lists.electorama.com/pipermail/election-methods-electorama.com/2015-January/131069.html) 
- [Second EM list post, simplifying Bucklin-Range](http://lists.electorama.com/pipermail/election-methods-electorama.com/2016-November/000976.html)

[1] Jain, Kamal, Mohammad Mahdian, and Amin Saberi. "A new greedy approach for facility location problems." (2002). <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.29.2072>.

## Description

The scripts may be somewhat ugly as they're meant as proofs of concept and
some of the tiebreak logic isn't entirely sound yet.

- `bucklin.py` implements the weighted candidate version of Bucklin-Range 
in both Bucklin ("Double Bucklin") and Range-based forms. This is essentially
party list with a limited number of parties, but where the parliament has
infinite seats and each candidate's weight is the proportion of the seats that
candidate would get if it were a party.

- `dbuck_partylist.py` implements a discrete version of Double Bucklin that
reduces to Webster's method if the voters only submit Plurality-style ballots.
To reduce vote-wasting, the method only gives additional seats to a party when
there are enough voters to give one additional seat, and then it redistributes
the surplus to the voters who contributed to giving the party an additional
seat (as in STV).

- `dbuck_droop.py` is a simple Double Bucklin variant which elects 
individual candidates (no weights). It is basically Bucklin or MJ with the 
Droop quota (rather than a majority) as the threshold.
