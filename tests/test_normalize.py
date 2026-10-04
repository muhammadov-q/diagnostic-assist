import unicodedata
import unittest

from diagnostic_assist.domain import PartRole
from diagnostic_assist.labeling.normalize import (
    PartCatalogue,
    extract_error_codes,
    fold,
    normalise_error_code,
    part_family,
)


class ErrorCodeTest(unittest.TestCase):
    def test_variants_of_one_code_normalise_to_the_same_value(self):
        for raw in ["E-207", "E207", "err 207", "Error: 207", "Fehlercode 207", "errore 207"]:
            with self.subTest(raw=raw):
                self.assertEqual(normalise_error_code(raw), "E207")

    def test_other_prefixes_are_kept(self):
        self.assertEqual(normalise_error_code("LP-01"), "LP01")

    def test_free_text_scan_finds_codes_and_ignores_model_names_and_measurements(self):
        text = "CX-450 overheats, E-207 on panel; 12.1V at 1500rpm, 118C, R410A, cyl 1. Display flashed err 207 once."
        self.assertEqual(extract_error_codes(text, exclude_prefixes={"CX"}), {"E207"})

    def test_words_after_error_are_not_code_prefixes(self):
        self.assertEqual(extract_error_codes("error at 300 hours, Fehler bei 200 bar"), set())


class PartTest(unittest.TestCase):
    def test_part_family_strips_the_equipment_model(self):
        self.assertEqual(part_family("SEALKIT-BC-120", "AP-120"), "SEALKIT-BC")
        self.assertEqual(part_family("UNIT-CX450", "CX-450"), "UNIT")
        self.assertEqual(part_family("REED-CX", "CX-300"), "REED-CX")
        self.assertEqual(part_family("BATT-95AH", "CX-450"), "BATT-95AH")

    def test_catalogue_knows_whole_units_and_consumables(self):
        catalogue = PartCatalogue()
        self.assertIs(catalogue.role_hint("UNIT-CX450"), PartRole.WHOLE_UNIT)
        self.assertIs(catalogue.role_hint("REFRIG-R410A"), PartRole.CONSUMABLE)
        self.assertIsNone(catalogue.role_hint("CONTACTOR-M1"))


class FoldTest(unittest.TestCase):
    def test_umlauts_accents_case_and_punctuation(self):
        self.assertEqual(fold("Hauptschütz defekt."), fold("Hauptschuetz defekt"))
        self.assertEqual(fold("Vérin rayé,"), "verin raye")

    def test_decomposed_unicode_folds_like_composed(self):
        self.assertEqual(fold(unicodedata.normalize("NFD", "Hauptschütz")), fold("Hauptschütz"))


if __name__ == "__main__":
    unittest.main()
