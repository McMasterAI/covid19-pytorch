import unittest

import numpy as np


class TestPreprocess(unittest.TestCase):
    def test_rolling_mean(self):
        from preprocess import rolling_mean

        window = 3
        mylist = [1, 2, 3, 4, -5, 0]
        expected = [
            1 / 1,
            (1 + 2) / 2,
            (1 + 2 + 3) / 3,
            (2 + 3 + 4) / 3,
            (3 + 4 + -5) / 3,
            (4 + -5 + 0) / 3,
        ]
        self.assertEqual(len(mylist), len(expected))  # sanity check

        means = rolling_mean(mylist, window=window)
        self.assertEqual(len(mylist), len(means))
        self.assertTrue(np.allclose(means, expected))


if __name__ == "__main__":
    unittest.main()
