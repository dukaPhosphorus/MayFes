# -*- coding: utf-8 -*-
#
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import os
import time

import subprocess


backnumber_dir = './back_number'
bach_dir = './Bach'
ARASHI_dir = './ARASHI'


class ChangeHandler_back_number(FileSystemEventHandler):

    def __init__(self):
        observer = Observer()
        observer.schedule(self, backnumber_dir, recursive=True)
        observer.start()

    def on_created(self, event): #target_dirに追加された時
        filepath = event.src_path
        filename = os.path.abspath(filepath)
        print(backnumber_dir + 'に%sが作成されました' % filename)

        print('deepbacknumber.py 実行')
        os.chdir('./Deepbacknumber_tsteps_32_light')
        #filepath = './' + filepath[2:]
        cmd = 'python ./deepbacknumber.py -i 1500 -m ' + filename  #### 8小節　-i 500 で生成30秒
        subprocess.call(cmd.split())
    
        time.sleep(2.0) #楽譜表示5秒後に作ってもらったmididata削除
        os.chdir('..')
        os.remove(filepath) 

    def on_deleted(self, event):  #target_dirに削除された時
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print(backnumber_dir + 'から%sが削除されました' % filename)



class ChangeHandler_ARASHI(FileSystemEventHandler):

    def __init__(self):
        observer = Observer()
        observer.schedule(self, ARASHI_dir, recursive=True)
        observer.start()

    def on_created(self, event): #target_dirに追加された時
        filepath = event.src_path
        filename = os.path.abspath(filepath)
        print(ARASHI_dir + 'に%sが作成されました' % filename)

        print('deeparashi.py 実行')
        os.chdir('./DeepARASHI_jikken_1')
        #filepath = './' + filepath[2:]
        cmd = 'python ./deeparashi.py --ext lstm100 -i 1500 -m ' + filename   #### 8小節　-i 500 で生成11秒
        subprocess.call(cmd.split())
    
        time.sleep(2.0) #楽譜表示5秒後に作ってもらったmididata削除
        os.chdir('..')
        os.remove(filepath) 

    def on_deleted(self, event):  #target_dirに削除された時
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print(ARASHI_dir + 'から%sが削除されました' % filename)



class ChangeHandler_Bach(FileSystemEventHandler):

    def __init__(self):
        observer = Observer()
        observer.schedule(self, bach_dir, recursive=True)
        observer.start()

    def on_created(self, event): #target_dirに追加された時
        filepath = event.src_path
        filename = os.path.abspath(filepath)
        print(bach_dir + 'に%sが作成されました' % filename)

        print('deepBach.py 実行')
        #subprocess.call(["cd", "DeepBach"], shell=True)
        os.chdir('./DeepBach')
        #filepath = './' + filepath[2:]
        cmd = 'python ./deepBach.py -i 5000 -m ' + filename   ### 8小節 -i 5000 で生成55秒
        subprocess.call(cmd.split())
    
        time.sleep(2.0) #楽譜表示5秒後に作ってもらったmididata削除
        os.chdir('..')
        os.remove(filepath) 

    def on_deleted(self, event):  #target_dirに削除された時
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print(bach_dir + 'から%sが削除されました' % filename)







if __name__ in '__main__':
    while 1:
        #event_handler = ChangeHandler()
        #observer = Observer()
        #observer.schedule(event_handler, backnumber_dir, recursive=True)
        #observer.start()
        ChangeHandler_back_number()
        ChangeHandler_ARASHI()
        ChangeHandler_Bach()

                   
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
