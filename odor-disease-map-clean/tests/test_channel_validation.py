import json, sys, tempfile, unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
import channel_validation as cv


class ChannelValidationTests(unittest.TestCase):
    def setUp(self):
        self.ids, self.x, self.channels = cv.synthetic_table(seed=1)

    def test_grouped_split_is_not_more_optimistic_than_pixel_split(self):
        y = self.ids["analyte"]
        classes = list(np.unique(y))
        leaky = cv.summarise(y, cv.random_pixel_split(self.x, y), classes)["accuracy"]
        film = cv.summarise(y, cv.leave_one_group_out(self.x, y, self.ids["film"]), classes)["accuracy"]
        self.assertGreaterEqual(leaky, film)
        self.assertGreater(film, 1 / len(classes))

    def test_variance_fractions_sum_to_one(self):
        sel = self.ids["analyte"] == "indole_A"
        d = cv.variance_decomposition(self.x[sel, 0], self.ids["film"][sel], self.ids["map"][sel])
        self.assertAlmostEqual(d["between_film"] + d["between_map_within_film"] + d["within_map"], 1.0, places=9)
        self.assertGreater(d["between_film"], 0.0)

    def test_single_channel_weaker_than_joint(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = cv.run(self.ids, self.x, self.channels, tmp, make_plots=False)
            # MPY and NT each separate one pair only; the joint pattern separates all three
            joint = report["validation"]["leave_one_film_out"]["accuracy"]
            self.assertGreater(joint, max(report["single_channel_grouped_accuracy"].values()) - 1e-9)
            self.assertTrue((Path(tmp) / "channel_validation.json").is_file())

    def test_csv_roundtrip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pixels.csv"
            cv.write_table(p, self.ids, self.x, self.channels)
            ids, x, channels = cv.read_table(p)
            self.assertEqual(channels, self.channels)
            np.testing.assert_allclose(x, self.x, rtol=1e-5)
            bad = Path(tmp) / "bad.csv"
            bad.write_text("film,map,MPY\nf,m,1\n")
            with self.assertRaises(ValueError):
                cv.read_table(bad)

    def test_holding_out_only_class_fails_loudly(self):
        y = np.array(["a"] * 4 + ["b"] * 4)
        groups = np.array(["g1"] * 4 + ["g2"] * 4)
        x = np.arange(8, dtype=float)[:, None]
        with self.assertRaises(ValueError):
            cv.leave_one_group_out(x, y, groups)


if __name__ == "__main__":
    unittest.main()
