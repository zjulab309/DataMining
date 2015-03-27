# Data Mining Main Function

# argparse is a standard Python mechanism for handling commandline
# args while avodiding a bunch of boilerplates code
import argparse

from utils import fs

def main():
    # Build the commandline parser and return entered args.
     args = configure_command_line_arguments()

     fs.read_csv(args['file'])
     print args

################################################################################
#
# Build the commandline parser for the script and return a map for the entered
# options.  In addition, setup logging based on the user's entered log level.
# Specific options are documented inline.
#
################################################################################

def configure_command_line_arguments():
    # Initialize the commandline arguments parser
    parser = argparse.ArgumentParser(description= 'Data Mining Platform.')

    # Configure the log level parser.  Verbose shows some logs, veryVerbose show
    # more detailed log
    logging_group = parser.add_mutually_exclusive_group(required = False)
    logging_group.add_argument("-v",
                               "--verbose",
                               help = "Set the log level verbose",
                               action = "store_true",
                               required = False)

    logging_group.add_argument("-vv",
                               "--veryVerbose",
                               help = "Set the log level very verbose",
                               action = 'store_true',
                               required = False)

    # Tell the parser the route of the data file.
    parser.add_argument("-f",
                        "--file",
                        help = "Tell the path of the data file and this parameter can't be ignored!",
                        required = True)

    # Parse the passed commandline args and turn then into a dictionary
    args = vars(parser.parse_args())

    return args

################################################################################
# This is a pythonsim.
################################################################################
if __name__ == "__main__":
    # execute main() function
    main()
