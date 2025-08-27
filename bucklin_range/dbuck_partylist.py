# Double Bucklin party list (Webster-based)

import numpy as np
from scipy.optimize import bisect

lcr_names = ["Left", "Center", "Right"]
lcr = np.array([
	[10, 5, 0],
	[0, 5, 10],
	[6, 10, 0],
	[0, 10, 5]
	])

lcr_weights = [3, 3, 1, 1]

# https://en.wikipedia.org/wiki/Webster/Sainte-Lagu%C3%AB_method
sainte_seats = 7
sainte_weights = [53, 24, 23]
sainte_names = ["Party A", "Party B", "Party C"]
sainte = np.array([
	[10, 0, 0],
	[0, 10, 0],
	[0, 0, 10]])

def thresholded_party_list(vote_array, vote_weights_in, cand_names, 
	threshold, min_seats=1, verbose=True):

	# The first seat threshold being so low in Sainte-Laguë compared to the
	# thresholds for seats 2, 3, ..., means there's an incentive for parties
	# to split. In a real setting, for this reason, min_seats should 
	# probably be at least 2. Alternatively, setting minimum_threshold =
	# threshold will give something more like D'Hondt or modified SL.
	minimum_threshold = (2*min_seats - 1) * threshold/2.0

	# we're going to modify the ballot weights
	vote_weights = np.copy(vote_weights_in).astype(float) 

	numvoters = np.sum(vote_weights)
	numballots = len(vote_array)
	numcands = len(vote_array[0])
	rmax = np.max(vote_array)

	bar = rmax

	C = set(xrange(numcands))
	unelected_C = set(xrange(numcands))		# Candidates still in the running
	elected_C = set()						# Candidates not still in the...
	total_support = np.array([0.0] * numcands)
	seats = np.array([0] * numcands)

	while bar > 0:

		something_happened = False

		# If an unelected party has more voters above the bar than 
		# minimum_threshold, consider it electable.
		electable = []
		for cand in unelected_C:
			cand_score = sum([vote_weights[v] for v in xrange(numballots)
								if vote_array[v][cand] >= bar])

			if cand_score >= minimum_threshold:
				electable.append((cand_score, cand))

		# Choose the electable party with the greatest score, if any,
		# and elect it. (A bit different from the weighted vote version.)
		if len(electable) != 0:
			to_elect_score, to_elect = sorted(electable, reverse=True)[0]

			unelected_C.remove(to_elect)
			elected_C.add(to_elect)
			something_happened = True
			total_support[to_elect] += minimum_threshold
			seats[to_elect] += min_seats

			if verbose:
				print "Electing", cand_names[to_elect], "mt=", minimum_threshold, \
					"actual support=", to_elect_score, "seats=", min_seats, \
					"bar=", bar

			# Redistribute the surplus by reweighting everybody who
			# contributed to getting the winner elected.
			surplus = to_elect_score - minimum_threshold
			reweight = surplus/float(to_elect_score)
			
			for v in xrange(numballots):
				if vote_array[v][to_elect] >= bar:
					vote_weights[v] *= reweight

			something_happened = True

		# If an elected party is eligible for another seat, allocate that
		# seat and redistribute votes.
		for cand in elected_C:
			cand_score = sum([vote_weights[v] for v in xrange(numballots)
							if vote_array[v][cand] >= bar])

			if cand_score >= threshold:
				something_happened = True
				total_support[cand] += threshold
				seats[cand] += 1

				surplus = cand_score - threshold
				reweight = surplus/float(cand_score)

				for v in xrange(numballots):
					if vote_array[v][cand] >= bar:
						vote_weights[v] *= reweight

				if verbose:
					print "Giving another seat to", cand_names[cand], "t=", \
					threshold, "actual support=", cand_score, "tot seats=", \
					seats[cand], "bar=", bar

		if not something_happened:
			bar -= 1	# one rating point

	# total_support must be altered somehow...
	if verbose:
		print "seats", seats
		print "total support", total_support

	# We add 1e-6 because of precision problems. The calculation below
	# shows that the number of seats we got matches what you'd get using 
	# ordinary Webster on the total support array with the threshold as the 
	# divisor.
	return np.round(1e-6+total_support/threshold)

def get_least_threshold(vote_array, vote_weights_in, cand_names, num_winners, verbose=True):
	def distance_to_desired_num_winners(x):
		num_winners_with_x = sum(thresholded_party_list(vote_array, 
			vote_weights_in, cand_names, x, verbose=False))
		# Set the output so that we have the root f(x)=0 when the
		# number of winners is equal to the desired number of winners
		distance = num_winners - num_winners_with_x
		if verbose:
			print "Desired num winners", num_winners, "actual:", num_winners_with_x, "x:", x
		# and bias it slightly so we get the least threshold where that
		# is true.
		return distance + 1.0/(x+2)

	max_threshold = sum(vote_weights_in)
	around_root = bisect(distance_to_desired_num_winners, 0.001, max_threshold)

	return around_root

print "==== LCR ===="
print thresholded_party_list(lcr, lcr_weights, lcr_names, 10)
print "==== LCR, single seat ===="
print thresholded_party_list(lcr, lcr_weights, lcr_names, 5)

print "==== Wikipedia Sainte-Lague example ===="
print get_least_threshold(sainte, sainte_weights, sainte_names, sainte_seats)
print thresholded_party_list(sainte, sainte_weights, sainte_names, 
	15 + 1/7.0 + 1e-6)