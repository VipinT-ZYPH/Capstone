import unittest
from pathlib import Path

from voiceops.dataset import DatasetPaths, DatasetRepository, business_days_after


class DatasetFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.repository = DatasetRepository(DatasetPaths.discover(root))

    def test_supplied_practice_counts(self):
        self.assertEqual(len(self.repository.practice_questions()), 52)
        self.assertEqual(len(self.repository.practice_conversations()), 96)

    def test_sections_preserve_ids_and_metadata(self):
        sections = self.repository.sections()
        self.assertGreater(len(sections), 0)
        self.assertTrue(all(section.section_id for section in sections))
        self.assertTrue(all(section.metadata["status"] for section in sections))

    def test_business_day_calendar(self):
        holidays = {__import__("datetime").date(2026, 10, 19)}
        self.assertEqual(
            business_days_after(__import__("datetime").date(2026, 10, 17), 1, holidays),
            __import__("datetime").date(2026, 10, 20),
        )


if __name__ == "__main__":
    unittest.main()
