# Double Bucklin candidate method (Droop quota)

# Should pass monotonicity and something analogous to the Droop quota for
# graded (MJ-style) ballots.

# Vulnerable to vote management.

# TODO: Use rational numbers here to find out whether the "true" Droop quota
# missing really is the result of numerical instability or a more fundamental
# bug.

import numpy as np

lcr_names = ["Left", "Center", "Right"]
lcr = np.array([
	[10, 5, 0],
	[0, 5, 10],
	[6, 10, 0],
	[0, 10, 5]
	])

lcr_weights = [3, 3, 1, 1]

# http://www.rangevoting.org/STVPRunger.html
unger_names = ["A", "B", "C", "D", "E", "F"]
unger = np.array([
	 #A #B #C #D #E #F
	[4, 3, 2, 1, 0, 0],		#A>B>C>D
	[0, 4, 0, 3, 2, 1],		#B>D>E>F
	[3, 0, 2, 0, 0, 4],     #F>A>C
	[0, 0, 3, 4, 2, 1],		#D>C>E>F
	[0, 2, 4, 3, 0, 0]		#C>D>B
	])

unger_weights = [20, 20, 20, 20, 19]

def elect(vote_array, vote_weights_in, cand_names, num_seats, 
	verbose=True):

	vote_weights = np.copy(vote_weights_in).astype(float) 

	numvoters = np.sum(vote_weights)
	numballots = len(vote_array)
	numcands = len(vote_array[0])
	rmax = np.max(vote_array)

	bar = rmax

	# I think we need to add just a little bit for numerical instability
	# purposes, but I'm not sure about that.

	droop_quota = numvoters/float(num_seats+1) 
	droop_quota += 1e-3/numvoters				# Numerical instability???

	unelected_C = set(xrange(numcands))		# Candidates still in the running
	elected_C = set()						# Candidates not still in the...

	while bar > 0:
		something_happened = False

		# If an unelected candidate has at least a Droop quota's worth of 
		# votes, elect him.

		electable = []
		for cand in unelected_C:
			cand_score = sum([vote_weights[v] for v in xrange(numballots)
								if vote_array[v][cand] >= bar])

			if cand_score >= droop_quota:
				electable.append((cand_score, cand))

		# Choose the electable candidate with the greatest score, if any,
		# elect him, and redistribute the surplus.
		if len(electable) != 0:
			to_elect_score, to_elect = sorted(electable, reverse=True)[0]

			unelected_C.remove(to_elect)
			elected_C.add(to_elect)
			something_happened = True

			if verbose:
				print "Electing", cand_names[to_elect], "\tDQ=", droop_quota,\
					"actual support=", to_elect_score, 	"bar=", bar

			# Redistribute the surplus by reweighting everybody who
			# contributed to getting the winner elected.
			surplus = to_elect_score - droop_quota
			reweight = surplus/float(to_elect_score)
			
			for v in xrange(numballots):
				if vote_array[v][to_elect] >= bar:
					vote_weights[v] *= reweight

			something_happened = True

		if not something_happened:
			bar -= 1

	return [cand_names[x] for x in elected_C]

print elect(lcr, lcr_weights, lcr_names, 2)
print elect(lcr, lcr_weights, lcr_names, 1)

print elect(unger, unger_weights, unger_names, 3)