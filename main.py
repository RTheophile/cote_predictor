import os
import pathlib
import nba_cote_computer as ncc

def main():
    # set the script directory as the working directory
    os.chdir(pathlib.Path(__file__).parent.absolute())
    
    # Launch all the work
    cote_computer = ncc.CoteComputer()
    cote_computer.execute()

if __name__ == "__main__":
    main()
    