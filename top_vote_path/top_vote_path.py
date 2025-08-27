# Implement the monotone resistant set method, tentatively labeled
# "Top Vote Path" (TVP) for its use of paths of first preferences over
# different subelections.
# The election is a list of ballots: each ballot is a list containing
# the weight of the ballot, followed by the candidates in ranked order.

# Equal-rank has not been implemented. One way to do so would be to use
# fractional rank (where D>A=B>C gives half a point to A and B in the
# subelection {A, B, C}), but this produces surprising outcomes in
# certain cases.
# If a ballot equal-ranks every candidate in a subelection, it should
# be considered exhausted and thus ignored.

# Truncation should work, but the resulting method fails Plurality.

# The method should be monotone, as should Condorcet//TVP and Smith,TVP.
# (I haven't proven the Smith case but I haven't found a counterexample
# either.)

import numpy as np

# Generate a list of lists of n True/False entries corresponding to the
# number of ways that one can pick at least one element from a set of
# n elements. This is the binary expansion of 1...n, inclusive.

def get_binary_nonempty_power_set(n):
	sets = []

	for i in range(1, 2**n):
		# Add each bit in i from most to least significant.
		sets.append([bool(i & 2**(n-bit-1)) for bit in range(n)])

	return sets

# Election: The input election.
# Subelections: A list of relevant subelections in binary set form.

# It might be possible to optimize this counting procedure by using
# an ordinary power set instead of a binary representation, but it's
# harder to code, so I've chosen the easier-to-understand approach here.
def count_subelection_first_prefs(election, subelections):
	num_subelections = len(subelections)
	num_candidates = len(subelections[0])

	first_prefs = np.zeros((num_subelections, num_candidates))

	for subelection_idx in range(num_subelections):
		for weight, ballot in election:
			for candidate in ballot:
				# If the candidate at this rank of the ballot is
				# in the subelection, add the weight to his first
				# preference count.
				if subelections[subelection_idx][candidate]:
					first_prefs[subelection_idx][candidate] += weight
					break

	# Get the number of voters who expressed a preference in each
	# subelection. If the ballots are complete, this should just
	# be the number of voters.
	subelections_voters = np.sum(first_prefs, axis=1)

	return first_prefs, subelections_voters

# If the non_leaves list is A_1, A_2, ..., A_n, that means that
# the path A_1->A_2->...->A_n has been established as valid. If
# L is the leaf and A_1->...->A_n->L is valid, then we recurse
# on candidates from there; otherwise we'll stop here.
def recurse_path(non_leaves, leaf, subelections, first_prefs,
	subelections_voters, has_path_to, is_on_path):

	if len(non_leaves) == 0:
		raise ValueError("recurse_path: Must have at "
			"least one non-leaf.")

	num_subelections = len(subelections)
	num_candidates = len(subelections[0])

	for subelection_idx in range(num_subelections):
		subelection = subelections[subelection_idx]
		subelection_first_prefs = first_prefs[subelection_idx]
		subelection_voters = subelections_voters[subelection_idx]

		# The subelection must contain the last non-leaf and
		# the leaf to say anything about whether the last
		# non-leaf indirectly disqualifies the laf. If the
		# subelection doesn't contain them, skip it.
		if not subelection[leaf] or not subelection[non_leaves[-1]]:
			continue

		# Number of candidates: true counts as 1, false as 0
		subelection_cands = sum(subelection)

		# Find the first non-leaf candidate in the subelection
		# by path index. This must exist since we know that
		# at least the last non-leaf is in the subelection.
		for i in range(len(non_leaves)):
			if subelection[non_leaves[i]]:
				first_inside_idx = i
				break

		# Check that every subsequent candidate in the path is
		# also in the subelection. If not, the subelection doesn't
		# have a valid suffix set and thus imposes no constraint.
		valid_suffix_set = True
		for non_leaf in non_leaves[first_inside_idx:]:
			valid_suffix_set &= subelection[non_leaf]

		if not valid_suffix_set:
			continue		# TEST NEEDED: detect someone putting "break" here

		# Get the sum of first preferences over the
		# suffix set.
		Q_first_preferences = 0
		for non_leaf in non_leaves[first_inside_idx:]:
			Q_first_preferences += subelection_first_prefs[non_leaf]

		# Let Q be the largest suffix set, |S| be the number of
		# candidates in the subelection, and v_S then number of voters
		# expressing a preference between candidates in the subelection.
		# Then the subelection imposes the constraint that
		# (sum first prefs over candidates X in Q: fpX) > v_S * |Q|/|S|.

		# If we fail a constraint, that indicates that the path
		# we're trying to build is non-viable, so return without
		# recursing.
		suffix_set_length = len(non_leaves) - first_inside_idx

		if Q_first_preferences <= \
			subelection_voters * suffix_set_length/subelection_cands:
			return

	# Okay, we passed. Mark that we have a path from the root
	# to the leaf, and add the leaf to the path.
	has_path_to[leaf] = True
	is_on_path[leaf] = True
	non_leaves.append(leaf)
	# Recurse on every candidate that's not already on our path.
	for cand in range(num_candidates):
		if not is_on_path[cand]:
			recurse_path(non_leaves, cand, subelections,
				first_prefs, subelections_voters, has_path_to,
				is_on_path)

	# We're done recursing, so clean up the path before exiting.
	non_leaves.pop()
	is_on_path[leaf] = False
	return

# Returns a matrix where x[i][j] is true if i has a path to j.
def get_paths(subelections, first_prefs, subelections_voters):
	path_matrix = []
	num_candidates = len(subelections[0])

	for cand in range(num_candidates):
		has_path_to = [False] * num_candidates
		is_on_path = [False] * num_candidates

		for leaf_cand in range(num_candidates):
			# First skip some path comparisons that we know
			# can't happen. These skips don't change the
			# outcome, they only speed up the calculation.

			# 1. Irreflexivity makes A==>A impossible.
			if cand == leaf_cand:
				continue

			# 2. Antisymmetry makes A==>B impossible if
			#    we already have B==>A.
			if cand > leaf_cand and path_matrix[leaf_cand][cand]:
				continue

			# It's always easier to add candidates to a long
			# path than a short path. So if we have A==>B
			# then there's no need to check A->B, because
			# whenever A->B leads to C, so does A==>B.
			if has_path_to[leaf_cand]:
				continue

			recurse_path([cand], leaf_cand, subelections,
				first_prefs, subelections_voters, has_path_to,
				is_on_path)

		path_matrix.append(has_path_to)

	return path_matrix

def get_scores(election, num_candidates):
	# Calculate the subelections and auxiliary stats required.
	subelections = get_binary_nonempty_power_set(num_candidates)
	first_prefs, subelections_voters = count_subelection_first_prefs(
		election, subelections)

	# Get what candidates have paths to others.
	paths = get_paths(subelections, first_prefs, subelections_voters)

	# Calculate in(X) and out(X).
	in_vals = np.array(paths).sum(axis=0)
	out_vals = np.array(paths).sum(axis=1)

	# Create and return a score tuple for each candidate, where the
	# winner/s have maximum score. (minimum in(X) and, of these,
	# maximum out(X)).
	# (Tests seem to indicate that just using out(X) is good enough,
	# but I haven't verified that.)
	return list(zip(-in_vals, out_vals))

def get_winner_set(election, num_candidates):
	scores = get_scores(election, num_candidates)
	max_score = max(scores)

	# Get the max scorers and return them.
	return [i for i in range(num_candidates) if scores[i] == max_score]

# For comparison, this function returns the resistant set.
def get_resistant_set(election, num_candidates):
	subelections = get_binary_nonempty_power_set(num_candidates)
	first_prefs, subelections_voters = count_subelection_first_prefs(
		election, subelections)

	num_subelections = len(subelections)
	num_candidates = len(subelections[0])

	def disqualifies(challenger, defender):
		if challenger == defender:
			return False

		for subelection_idx in range(num_subelections):
			# Number of candidates: true counts as 1, false as 0
			subelection_cands = sum(subelections[subelection_idx])
			subelection_voters = subelections_voters[subelection_idx]
			subelection = subelections[subelection_idx]

			if not (subelection[challenger] and subelection[defender]):
				continue

			if not (first_prefs[subelection_idx][challenger] \
					> subelection_voters/subelection_cands):
				return False

		return True

	def disqualified(defender):
		return any((disqualifies(challenger, defender) for
			challenger in range(num_candidates)))

	return [i for i in range(num_candidates) if not disqualified(i)]

# Condorcet cycle example.
condorcet_cycle = [
	[65, [0, 1, 2]],
	[50, [1, 2, 0]],
	[45, [2, 0, 1]]]

# Different from IFPP.
# This method gives A>C>B, IFPP gives C>A>B.
differs_from_ifpp = [
	[3, [0, 1, 2]],
	[1, [1, 0, 2]],
	[1, [1, 2, 0]],
	[3, [2, 0, 1]]]

# Should give be C>D>B=A. IRV gives D>C>B>A.
differs_from_irv_four = [
	[1, [0, 3, 2, 1]],
	[1, [1, 3, 0, 2]],
	[1, [2, 0, 3, 1]],
	[1, [2, 1, 3, 0]],
	[1, [3, 1, 2, 0]]]

# Gives A>B>C, but DAC/DSC says B is the winner.
differs_from_dac = [
	[3, [0, 1, 2]], 
	[2, [0, 2, 1]],
	[1, [1, 0, 2]],
	[5, [1, 2, 0]],
	[2, [2, 0, 1]]]

# Should give a tie, A=B=C.
inconclusive = [
	[3, [0, 1, 2]],
	[2, [1, 0, 2]],
	[1, [1, 2, 0]],
	[3, [2, 1, 0]]]

# Detect incorrect recursion rules.
no_early_stopping = [
	[1, [0, 2, 4, 1, 3]],
	[2, [0, 4, 1, 2, 3]],
	[1, [1, 2, 3, 4, 0]],
	[1, [2, 1, 3, 4, 0]],
	[1, [2, 3, 0, 1, 4]],
	[1, [4, 1, 3, 2, 0]]]

# Test incorrectly using break in get_paths, has_path_to[leaf_cand].
mistaken_transitivity = [
	[2, [1, 0, 2, 3]],
	[1, [1, 3, 0, 2]],
	[2, [3, 1, 2, 0]]]

# "Naive" truncation fails the Plurality criterion. Example by Kevin
# Venzke.
plurality_failure = [
	[29, [0]],
	[28, [1,2]],
	[23, [2]],
	[20, [3,2]]]

tests_expected_winners_rs = [
	["Condorcet cycle", 3, condorcet_cycle, [0], [0, 2]],
	["Differs from IFPP", 3, differs_from_ifpp, [0], [0, 2]],
	["Differs from IRV, 4 cddts", 4, differs_from_irv_four, [2], [2, 3]],
	["Differs from DAC", 3, differs_from_dac, [0], [0]],
	["Inconclusive", 3, inconclusive, [0, 1, 2], [0, 1, 2]],
	["No early stopping", 5, no_early_stopping, [0], [0, 2]],
	["Mistaken transitivity", 4, mistaken_transitivity, [1], [1]],
	["Plurality failure", 4, plurality_failure, [1], [0, 1, 2]]]

for test_name, num_candidates, test_election, expected_outcome, \
	expected_resistant_set in tests_expected_winners_rs:

	seen_outcome = get_winner_set(test_election, num_candidates)
	if seen_outcome != expected_outcome:
		raise Exception(f"Winner: {test_name}: Got {seen_outcome}, "
			f"expected {expected_outcome}")

	seen_resistant_set = get_resistant_set(test_election, num_candidates)
	if seen_resistant_set != expected_resistant_set:
		raise Exception(f"Resistant set: {test_name}: Got {seen_resistant_set},"
			f"expected {expected_resistant_set}")

print("Tests OK")
