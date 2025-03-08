import time

from utils.car import Car
from utils.controller import Controller
from utils.dolphin import find_dolphin_dir


class DrivingTest:
    def __init__(self):
        self.dolphin_dir = find_dolphin_dir()
        if self.dolphin_dir is None:
            raise RuntimeError("Could not find dolphin config directory")

        self.ctrl_path = self.dolphin_dir + "/Pipes/pipe"
        self.controller = Controller(self.ctrl_path)
        self.controller.__enter__()
        self.car = Car(self.controller)

    def teardown(self):
        """Clean up resources"""
        if self.car:
            self.controller.reset()

        if self.controller:
            print("Resetting controller...")
            self.controller.reset()
            self.controller.__exit__()
            print("Controller reset complete")

    def test_basic_driving(self):
        """Basic driving test sequence"""
        print("\n=== Starting Basic Driving Test ===")

        # Short pause to prepare
        print("Starting in 3 seconds...")
        time.sleep(3)

        # Drive forward
        print("Testing forward driving")
        self.car.drive_forward()
        time.sleep(2.0)

        # Turn left
        print("Testing left turn")
        self.car.turn(0.2)
        time.sleep(1.5)

        # Turn right
        print("Testing right turn")
        self.car.turn(0.8)
        time.sleep(1.5)

        # Center and keep driving
        self.car.turn(0.5)
        time.sleep(1.0)

        # Stop
        self.car.stop_car()
        print("=== Basic Driving Test Complete ===\n")

    def test_drifting(self):
        """Test drifting mechanics"""
        print("\n=== Starting Drift Test ===")

        # Short pause to prepare
        print("Starting in 3 seconds...")
        time.sleep(3)

        # Start driving
        self.car.drive_forward()
        time.sleep(1.0)

        # Left drift
        print("Testing left drift")
        self.car.turn(0.2)
        time.sleep(0.5)
        self.car.drift()
        time.sleep(2.0)
        self.car.stop_drift()
        time.sleep(0.5)

        # Straightaway
        self.car.turn(0.5)
        time.sleep(1.0)

        # Right drift
        print("Testing right drift")
        self.car.turn(0.8)
        time.sleep(0.5)
        self.car.drift()
        time.sleep(2.0)
        self.car.stop_drift()

        # Reset
        self.car.turn(0.5)
        self.car.stop_car()
        print("=== Drift Test Complete ===\n")

    def test_item_usage(self):
        """Test using items"""
        print("\n=== Starting Item Usage Test ===")

        # Short pause to prepare
        print("Starting in 3 seconds...")
        time.sleep(3)

        # Start driving
        self.car.drive_forward()
        time.sleep(1.0)

        # Use item
        print("Using item")
        self.car.use_item()
        time.sleep(1.0)

        # Stop
        self.car.stop_car()
        print("=== Item Usage Test Complete ===\n")

    def test_race_scenario(self):
        """Simulate a short race"""
        print("\n=== Starting Race Scenario Test ===")

        # Countdown
        print("Race starting in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("GO!")

        # Race start
        self.car.drive_forward()
        time.sleep(1.5)

        # First turn (left)
        print("First turn - left")
        self.car.turn(0.2)
        time.sleep(1.0)

        # Straightaway
        print("Straightaway")
        self.car.turn(0.5)
        time.sleep(2.0)

        # Second turn with drift (right)
        print("Second turn - right with drift")
        self.car.turn(0.8)
        time.sleep(0.5)
        self.car.drift()
        time.sleep(1.5)
        self.car.stop_drift()
        time.sleep(0.5)

        # Use item
        print("Using power-up")
        self.car.use_item()
        time.sleep(0.5)

        # Final turn (left)
        print("Final turn - left")
        self.car.turn(0.3)
        time.sleep(1.0)

        # Final straightaway
        print("Final sprint!")
        self.car.turn(0.5)
        time.sleep(2.0)

        # Finish
        print("Finish line!")
        self.car.stop_car()
        print("=== Race Scenario Test Complete ===\n")

    def run_all_tests(self):
        """Run all driving tests"""
        try:
            print("\nReady to run tests. Make sure Dolphin is running.")
            input("Press Enter to start tests...")

            self.test_basic_driving()
            time.sleep(1.0)

            self.test_drifting()
            time.sleep(1.0)

            self.test_item_usage()
            time.sleep(1.0)

            self.test_race_scenario()

            print("\nAll tests completed successfully!")

        except Exception as e:
            print(f"Error during tests: {e}")
        finally:
            self.teardown()


def main():
    """Main entry point for driving tests"""
    try:
        test = DrivingTest()
        test.run_all_tests()
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
    print("Testing complete.")


if __name__ == "__main__":
    main()
