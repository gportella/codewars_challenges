# Description

>> I'm almost sure my code works, but fails at some tests in Codewars. My assumption is that their tests are broken.

Given a series of reads, assemble a circular genome. Some read could contain one missmatch, but no `indels` or anything too complex. Not all sets of reads are comprehensive. The function should accept a bool which tells if the reads can have mismatches.

Getting a full fledge de Bruijn assembly is a pain, lots of things can happen, so I opted for a combined approach. All a bit hacky, but in my hands it sort of works.
