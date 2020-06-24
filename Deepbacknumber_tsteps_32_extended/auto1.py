# -*- coding: utf-8 -*-

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import os
import time

import subprocess


target_dir = "/Users/fujiterunari/Desktop/demo"


class ChangeHandler(FileSystemEventHandler):

    def __init__(self):
        self.observer = Observer()
        self.observer.schedule(self, target_dir, recursive=False)
        self.observer.start()
        self.observer.join()

        
    def on_created(self, event): #target_dirに追加された時
        filepath = event.src_path
        #print(type(filepath))
        filename = os.path.basename(filepath)
        print('%sが作成されました' % filename)

        print('deepbacknumber.py 実行')
        cmd = "python /Users/fujiterunari/Desktop/Deepbacknumber_tsteps_32_extended/deepbacknumber.py -i 250 -m " + filepath
        subprocess.call(cmd.split())
        
        time.sleep(5.0) #楽譜表示5秒後に作ってもらったmididata削除
        os.remove(filepath) 

    def on_modified(self, event): #target_dirに変更が加わった時
        filepath = event.src_path
        filename = os.path.basename(filepath)
        #print('%sを変更しました' % filename)

    def on_deleted(self, event):  #target_dirに削除された時
        filepath = event.src_path
        filename = os.path.basename(filepath)
        print('%sが削除されました' % filename)


if __name__ in '__main__':
    while 1:
        ChangeHandler()
       
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
