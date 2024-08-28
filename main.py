import sys
import numpy as np

def main():
    x=np.ones(6)
    print(x)
    if len(sys.argv) > 1:
        print("Argument passed:", sys.argv[1])
    else:
        print("No argument passed.")

if __name__ == "__main__":
    main()
