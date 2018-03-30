#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from composition_eval import computeQuartiles
from composition_eval import eval_on_file

class composition_eval_tests(unittest.TestCase):

    def setUp(self):
        self.original_file = "./eval/test_data/original.txt"
        self.composed_file = "./eval/test_data/composed.txt"

    def test_quartiles_1(self):
        ranks = [6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
        np.testing.assert_equal(computeQuartiles(ranks), [15, 40, 43])

    def test_quartiles_2(self):
        ranks = [7, 15, 36, 39, 40, 41]
        np.testing.assert_equal(computeQuartiles(ranks), [15, 37.5, 40])

    def test_ranks(self):
        q1, q2, q3, word_ranks = eval_on_file(self.composed_file, self.original_file, None)
        self.assertEqual(word_ranks['märchenbuch'], 4)
        self.assertEqual(word_ranks['holzlöffel'], 5)
        self.assertEqual(word_ranks['gästehaus'], 1)

if __name__ == "__main__":
    unittest.main()
