import os
import unittest

from utils.controller import Button, Controller, Stick, Trigger


class ControllerTest(unittest.TestCase):
    def setUp(self):
        # Create a fifo for testing. We'll remove it in tearDown.
        self.fifo_path = os.getcwd() + "/fifo"
        os.mkfifo(self.fifo_path)
        self.pipe = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        self.controller = Controller(self.fifo_path)
        self.controller.__enter__()

    def tearDown(self):
        self.controller.__exit__()
        os.close(self.pipe)
        os.unlink(self.fifo_path)

    # Returns whatever is pending in the fifo
    def read_pipe(self):
        return os.read(self.pipe, 2048)

    def test_buttons_basic(self):
        self.controller.press_button(Button.A)
        self.assertEqual(self.read_pipe(), b"PRESS A\n")

    def test_buttons_multi(self):
        self.controller.press_button(Button.L)
        self.controller.press_button(Button.R)
        self.assertEqual(self.read_pipe(), b"PRESS L\nPRESS R\n")

    def test_buttons_release(self):
        self.controller.release_button(Button.A)
        self.assertEqual(self.read_pipe(), b"RELEASE A\n")

    def test_buttons_assert(self):
        with self.assertRaises(AssertionError):
            self.controller.press_button("Not a button")
        with self.assertRaises(AssertionError):
            self.controller.press_button(Trigger.L)

    def test_triggers_basic(self):
        self.controller.press_trigger(Trigger.L, 0)
        self.controller.press_trigger(Trigger.R, 1.0)
        self.assertEqual(self.read_pipe(), b"SET L 0.00\nSET R 1.00\n")

    def test_triggers_round(self):
        self.controller.press_trigger(Trigger.L, 0.124)
        self.controller.press_trigger(Trigger.R, 0.126)
        self.assertEqual(self.read_pipe(), b"SET L 0.12\nSET R 0.13\n")

    def test_triggers_assert(self):
        with self.assertRaises(AssertionError):
            self.controller.press_trigger(Button.L, 0.5)
        with self.assertRaises(AssertionError):
            self.controller.press_trigger(Trigger.R, 1.2)
        with self.assertRaises(AssertionError):
            self.controller.press_trigger(Trigger.R, -0.3)

    def test_sticks_basic(self):
        self.controller.tilt_stick(Stick.MAIN, 0, 0)
        self.assertEqual(self.read_pipe(), b"SET MAIN 0.00 0.00\n")
        self.controller.tilt_stick(Stick.C, 1, 0)
        self.assertEqual(self.read_pipe(), b"SET C 1.00 0.00\n")
        self.controller.tilt_stick(Stick.MAIN, 0.05, 0.55)
        self.assertEqual(self.read_pipe(), b"SET MAIN 0.05 0.55\n")

    def test_sticks_round(self):
        self.controller.tilt_stick(Stick.MAIN, 0.054, 0)
        self.assertEqual(self.read_pipe(), b"SET MAIN 0.05 0.00\n")
        self.controller.tilt_stick(Stick.MAIN, 0.056, 0)
        self.assertEqual(self.read_pipe(), b"SET MAIN 0.06 0.00\n")

    def test_sticks_assert(self):
        with self.assertRaises(AssertionError):
            self.controller.tilt_stick("Not a stick", 0.5, 0.5)
        with self.assertRaises(AssertionError):
            self.controller.tilt_stick(Stick.MAIN, -0.5, 0.5)
        with self.assertRaises(AssertionError):
            self.controller.tilt_stick(Stick.C, 0.5, 1.5)

    def test_reset(self):
        self.controller.reset()
        output = str(self.read_pipe())
        for button in Button:
            self.assertTrue("RELEASE {}".format(button.name) in output)
        for trigger in Trigger:
            self.assertTrue("SET {} 0.00".format(trigger.name) in output)
        for stick in Stick:
            self.assertTrue("SET {} 0.50 0.50".format(stick.name) in output)


if __name__ == "__main__":
    unittest.main()
