#function to write names of files in a directory to a text file

import os

def list_files(dir_path , file_name):
    '''function to write names of files in a directory to a text file'''
    with open(file_name , 'w') as f:
        for file in os.listdir(dir_path):
            f.write(file + '\n')

def list_dirs(root):
    for dir_name in os.listdir(root):
        list_files(os.path.join(root , dir_name) , dir_name + '.txt')

if __name__ == '__main__':
    list_dirs('/home/cvblgita/tri/Datasets/knowledge_distillation')
    

