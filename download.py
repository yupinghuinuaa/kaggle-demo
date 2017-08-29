import os
import subprocess
import concurrent.futures


SAVE_FOLDER = '/home/liuwen/ssd/kaggle'


def parser_urls(files='urls.txt'):
    name_ulrs = []
    with open(files, 'r') as f:
        for line in f.readlines():
            name, url = line.split()
            name_ulrs.append((name, url))

    return name_ulrs


def download_file(name, url):
    if str.startswith(name, 'train'):
        save_folder = os.path.join(SAVE_FOLDER, 'training')
    elif str.startswith(name, 'test'):
        save_folder = os.path.join(SAVE_FOLDER, 'testing')
    else:
        save_folder = os.path.join(SAVE_FOLDER, 'labels')

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    save_path = os.path.join(save_folder, name)

    # wget "urls"   -P  xxx    -O xxx
    # urls: the remote file
    # -P: the folder to save file
    # -O: the path of save file
    # wget "https:www.baidu.com/index.html" -P /home/liuwen/ -O /home/liuwen/baidu.html
    shell_commands = 'wget "%s" -P %s -O %s' % (url, save_folder, save_path)
    subprocess.run(shell_commands, shell=True, check=True)


def multi_process_download(name_urls):
    names, urls = [], []
    for name, url in name_urls:
        names.append(name)
        urls.append(url)

    with concurrent.futures.ProcessPoolExecutor() as pool:
        pool.map(download_file, names, urls)


if __name__ == '__main__':
    name_urls = parser_urls(files='urls.txt')
    multi_process_download(name_urls)


