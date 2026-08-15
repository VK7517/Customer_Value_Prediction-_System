import os 
import pickle 

def save_object(file_path,obj):

    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory,exist_ok=True)

    with open(file_path, "wb") as file:
        pickle.dump(obj,file)


def load_object(file_path):

    with open(file_path,"rb") as file:
        return pickle.load(file)