import unittest
from koboldcpp import replace_last_in_string

class TestStringUtils(unittest.TestCase):
    def test_replace_last_in_string_basic(self):
        self.assertEqual(replace_last_in_string("hello world", "world", "there"), "hello there")
        self.assertEqual(replace_last_in_string("apple apple apple", "apple", "orange"), "apple apple orange")

    def test_replace_last_in_string_not_found(self):
        self.assertEqual(replace_last_in_string("hello world", "abc", "def"), "hello world")

    def test_replace_last_in_string_empty_match(self):
        self.assertEqual(replace_last_in_string("hello world", "", "there"), "hello world")

    def test_replace_last_in_string_at_start(self):
        self.assertEqual(replace_last_in_string("apple orange", "apple", "banana"), "banana orange")

    def test_replace_last_in_string_at_end(self):
        self.assertEqual(replace_last_in_string("apple orange", "orange", "banana"), "apple banana")

    def test_replace_last_in_string_whole_string(self):
        self.assertEqual(replace_last_in_string("apple", "apple", "banana"), "banana")

    def test_replace_last_in_string_empty_text(self):
        self.assertEqual(replace_last_in_string("", "apple", "banana"), "")

    def test_replace_last_in_string_multiple_occurrences(self):
        self.assertEqual(replace_last_in_string("a-b-c-b-d", "b", "X"), "a-b-c-X-d")

    def test_replace_last_in_string_match_replacement_same(self):
        self.assertEqual(replace_last_in_string("hello world", "world", "world"), "hello world")

if __name__ == '__main__':
    unittest.main()
