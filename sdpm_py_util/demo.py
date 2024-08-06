import sys

def print_fn():
    print("Hi")

def sum_fn(a, b):
    print(int(a) + int(b))

if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])
