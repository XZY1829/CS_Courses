from pathlib import Path
import unittest

import extract_ppt


class ExtractPptPathTest(unittest.TestCase):
    def test_cpp_directory_follows_script_location(self) -> None:
        cpp_directory = getattr(extract_ppt, "cpp_directory", lambda: None)
        self.assertEqual(
            cpp_directory(),
            Path(extract_ppt.__file__).resolve().parent,
        )


if __name__ == "__main__":
    unittest.main()
