from Globals import SLEAPyException
from MainHelpers import ArgParser, ExperimentRunner, InputHandler
from ModelController.Modem.Modem_Controller import superMain

def main():
    """Main entry point"""
    try:
        # Get and apply args
        args = ArgParser.parse_arguments()
        
        # Get inputs (if needed)
        InputHandler.get_user_configuration(args)

        # Run experiment
        if args.polyarithmos:
            ExperimentRunner.run_experiment(polyarithmos=True)
            superMain()

        else:
            ExperimentRunner.run_experiment(polyarithmos=False)

    except SLEAPyException as e:
        print("Exception occured during run.")
        print(e)
    except Exception as e:
        raise SLEAPyException()


if __name__ == "__main__":
    superMain()
