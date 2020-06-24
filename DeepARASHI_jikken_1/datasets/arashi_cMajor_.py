from music21 import corpus, converter, stream, note, duration, analysis, interval, key, chord

from glob import glob

import os
import pickle


def main():
    file_list = glob('raw_dataset'+ '/*.xml')
    print(len(file_list))
    for k, file_name in enumerate(file_list):
        c = converter.parse(file_name)
        
        #print(file_name[len('original_dataset')+1:(len(file_name)-4)]) ####4=拡張子のlen+1
        
        chorale_key = c.analyze('key')
        sharp_num = chorale_key.sharps
        
        print(sharp_num,chorale_key)
        print(file_name)




if __name__ == '__main__':
    main()









    
