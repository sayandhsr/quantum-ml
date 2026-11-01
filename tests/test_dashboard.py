import unittest

from streamlit.testing.v1 import AppTest

from shared.preprocess import ROOT_DIR


class DashboardTests(unittest.TestCase):
    def test_dashboard_navigation_is_simple_and_emoji_free(self) -> None:
        app = AppTest.from_file(str(ROOT_DIR / "Dashboard" / "app.py"), default_timeout=30)
        app.run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        self.assertTrue(any("Diabetes classification" in item.value for item in app.title))
        self.assertEqual(len(app.tabs), 0)
        self.assertEqual(
            app.radio[0].options,
            [
                "Overview",
                "Dataset",
                "Traditional models",
                "Quantum model",
                "Enhanced Quantum (DR-QNN)",
                "Model comparison",
                "Try a prediction",
                "Method and limitations",
            ],
        )
        for page in app.radio[0].options:
            app.radio[0].set_value(page).run()
            self.assertEqual(
                len(app.exception),
                0,
                f"{page}: {[item.message for item in app.exception]}",
            )
        app.radio[0].set_value("Overview").run()
        overview_text = "\n".join(
            [item.value for item in app.markdown]
            + [item.value for item in app.subheader]
        )
        self.assertIn("Did the traditional model work?", overview_text)
        self.assertIn("Did the quantum model work?", overview_text)
        self.assertIn("correctly classified 74 of 100", overview_text)
        self.assertIn("correctly classified 55 of 100", overview_text)

    def test_saved_model_prediction_workflow(self) -> None:
        app = AppTest.from_file(str(ROOT_DIR / "Dashboard" / "app.py"), default_timeout=30)
        app.run()
        app.radio[0].set_value("Try a prediction").run()
        submit = next(
            button
            for button in app.button
            if button.label == "Calculate predictions"
        )
        submit.click().run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        available_models = list((ROOT_DIR / "traditional" / "models").glob("*.joblib"))
        if available_models:
            output = "\n".join(item.value for item in app.markdown)
            self.assertIn("Logistic Regression", output)
            self.assertIn("Quantum simulation", output)


if __name__ == "__main__":
    unittest.main()
