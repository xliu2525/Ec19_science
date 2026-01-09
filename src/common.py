from joblib import Memory
memory = None
root = None

def init(r):
    global memory
    global root
    root = r
    memory = Memory(root, verbose=0)