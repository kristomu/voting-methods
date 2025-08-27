import numpy as np
from scipy.optimize import bisect

# Very simple ballot format

lcr_names = ["Left", "Center", "Right"]
lcr = np.array([
	[10, 5, 0],
	[10, 5, 0],
	[10, 5, 0],
	[0, 5, 10],
	[0, 5, 10],
	[0, 5, 10],
	[6, 10, 0],
	[0, 10, 5]
	])

pizza_names = ["Pepperoni", "Mushroom"]
pizza = np.array([
	[10, 9],
	[10, 9],
	[0,  9]])

def thresholded(vote_array, cand_names, threshold, verbose=True, 
	bucklin_style=False):

	numvoters = len(vote_array)
	numcands = len(vote_array[0])

	V = set(xrange(numvoters))
	unassigned_V = set(xrange(numvoters))		# Set of all unassigned voters
	assigned_V = set() 				# Set of all voters assigned somewhere
	
	C = set(xrange(numcands))
	unelected_C = set(xrange(numcands))		# Candidates still in the running
	elected_C = set()						# Candidates not still in the...

	assigned_voters = [set() for x in xrange(numcands)] # voters assigned to cddts

	rmax = np.max(vote_array)		# maximum rating. WARNING: Assumes someone
									# votes max!

	b = rmax

	epsilon = 0.001

	while len(unassigned_V) > 0 and b >= 0:
		something_happened = False

		# Find potential supporters and electable candidates.

		electable = []
		for cand in unelected_C:
			potential_supporters = set([v for v in unassigned_V 
							if vote_array[v][cand] >= b])

			if bucklin_style:
				cand_score = len(potential_supporters)
			else:
				cand_score = sum([vote_array[s][cand] - b for s in 
					potential_supporters])

			if cand_score >= threshold:
				electable.append((cand_score, cand))

		if len(electable) != 0:
			# Take the first. TODO: Tie break. One potential tie break
			# could be to pick the one with most potential supporters,
			# because if we use QLTD logic (partially add in supporters)
			# then that candidate will clear the QLTD bar first; and after
			# that point, the case for elected candidates will add in all
			# the other supporters of that candidate. Possible further tie
			# break to pick the one with most potential supporters who are
			# also not supporters of some other candidate, or just use 
			# random voter hierarchy.

			# Something similar to this seems to be forced by the 
			# monotonicity requirement that lower thresholds should never 
			# give fewer winners. With a lower threshold, we must always 
			# elect someone who would be elected at this step with a higher
			# threshold, which is someone with a greater number of 
			# supporters (Bucklin) or a greater surplus sum (Range).

			# To see this, uncomment the line below and check pizza election
			# with 6.002 and 6.003.
			# to_elect = electable[0][1]

			# Choose the candidate with the greatest score, or the first
			# one if more than one has greatest score. Tiebreak would still
			# be needed for that part...
			to_elect = sorted(electable, reverse=True)[0][1]

			unelected_C.remove(to_elect)
			elected_C.add(to_elect)
			something_happened = True

			if verbose:
				print "There are now", len(electable), "electables."
				print "The electables list looks like this:", electable
				print "Elected ", cand_names[to_elect], "with b =", b

		# For every elected candidate, move his unassigned supporters to his
		# supporter list.
		for cand in elected_C:
			supporters = set([v for v in unassigned_V 
							if vote_array[v][cand] >= b])

			# Again, these will be stolen by the first candidate if there
			# are multiple. Again, a tie break could be nice.
			# Here I think that a fractional assignment could be useful,
			# where 1/n of each supporter's strength goes to the each
			# candidate.
			# Suppose we have
			# 10: A
			# 10: B
			# 1: C>A=B
			# then the reasonable two-seat result would be 10.5 voters to
			# A, 10.5 to B; not 11 to one and 10 to the other, since
			# they're perfectly symmetric. But that also suggests
			# 20: A=B
			# should give A and B 50% weight each, which in turn means
			# all electables should be elected at once above if everybody
			# who supports A also supports B. Such logic could get hairy
			# real fast with partly overlapping potential supporter sets.
			if len(supporters) > 0:
				assigned_voters[cand] |= supporters
				unassigned_V -= supporters
				something_happened = True

				if verbose:
					print "Giving voters", list(supporters), "to candidate",\
						cand_names[cand]

		# If nothing happened, slightly decrement b
		if not something_happened:
			b -= epsilon

	# Return winners and their weights.
	return [(cand_names[winner], len(assigned_voters[winner])) for winner in elected_C]

def get_least_threshold(vote_array, cand_names, num_winners, verbose=True, bucklin_style=False):
	
	def distance_to_desired_num_winners(x):
		num_winners_with_x = len(thresholded(vote_array, cand_names, x, 
			verbose=False, bucklin_style=bucklin_style))
		# Set the output so that we have the root f(x)=0 when the
		# number of winners is equal to the desired number of winners
		distance = num_winners - num_winners_with_x
		if verbose:
			print "Desired num winners", num_winners, "actual:", num_winners_with_x, "x:", x
		# and bias it slightly so we get the least threshold where that
		# is true.
		return distance + 1.0/(x+2)

	#distance_to_desired_num_winners = lambda x: len(thresholded(vote_array, 
	#	cand_names, x, verbose=False, bucklin_style=bucklin_style))-num_winners

	epsilon = 0.001
	numvoters = len(vote_array)
	rmax = np.max(vote_array)

	if bucklin_style:
		max_threshold = numvoters
	else:
		max_threshold = np.max(vote_array) # max rating

	# Select the x tolerance so that there's no chance the result can flip
	# between it and the true minimum.
	# The idea is that as we decrement the bar from rmax to 0, and after y
	# loopings, we reach 0. We want the difference between the true least
	# minimum bar_min and the root-finder's minimum to be less than epsilon
	# once we've gone y times around.
	# The solution to this is 
	#	tolerance = epsilon * (epsilon + max_threshold) / rmax
	# which we simplify as below, and multiply by 0.5 because we may be
	# off in both directions, and then again by 0.5 because we want to
	# be on the "too high threshold side" even though bisect may give us
	# a value just below the root or just above the root when the function
	# isn't continuous.
	x_tolerance = 0.25 * epsilon*epsilon/rmax
	around_root = bisect(distance_to_desired_num_winners, 0, max_threshold,
		xtol=x_tolerance)
	return around_root + x_tolerance

print "==== Pizza election, Range style ===="
#print thresholded(pizza, pizza_names, 6.002)
print thresholded(pizza, pizza_names, 6)
#print thresholded(pizza, pizza_names, 6.003)#, False)
#print 1/0
print "==== Pizza election, Bucklin style ===="
print thresholded(pizza, pizza_names, 1 + 1e-10, bucklin_style=True)
print "==== Too high a threshold gives Range style ===="
print thresholded(pizza, pizza_names, 3, False, True)

print "Approximately least threshold for pizza election:", \
	get_least_threshold(pizza, pizza_names, 1, False, False)
print "Approximately least threshold for Bucklin pizza:", \
	get_least_threshold(pizza, pizza_names, 1, False, True)

print "\n\n==== LCR, two winners ===="
print thresholded(lcr, lcr_names, 8.002)
print "==== LCR, one winner ===="
print thresholded(lcr, lcr_names, 22.0041)
print "LCR, two winners, Bucklin style, outcome only"
print "Note that the outcomes are the same as for Range style."
print thresholded(lcr, lcr_names, 2.001, False, bucklin_style=True)
print "LCR, one winner, Bucklin style, outcome only"
print thresholded(lcr, lcr_names, 4.001, False, bucklin_style=True)