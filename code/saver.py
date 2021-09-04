import os
import shutil
import time

from util import *

SRC = '../'
DST = '../archived/'
DIR_LIST = ['configs', 'curves', 'datasets', 'logs', 'models', 'results']
EXT_LIST = ['.json', '.txt', '.png', '.0', '.pth']
CFG = read_config()


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def save():
    path = f'{DST}{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}/'
    ensure_dir(path)
    for dir in DIR_LIST:
        print(f'Copying {SRC}{dir} to {path}{dir}...')
        ensure_dir(f'{path}{dir}')
        copytree(f'{SRC}{dir}', f'{path}{dir}')


def clear():
    for dir in DIR_LIST:
        if dir == 'configs' or dir == 'curves':
            continue
        for parent, dir_names, filenames in os.walk(f'../{dir}'):
            for fn in filenames:
                if os.path.splitext(fn.lower())[-1] in EXT_LIST:
                    print(f'Removing ../{dir}/{fn}...')
                    os.remove(os.path.join(parent, fn))


if __name__ == '__main__':
    mode = ''
    while mode != 'copy' and mode != 'move':
        print('Choose mode (copy/move):')
        mode = input()
        if mode == 'move':
            print('This option will delete existing files. Confirm? (yes/no)')
            confirm = input()
            if confirm != 'yes':
                mode = ''
    save()
    if mode == 'move':
        clear()
