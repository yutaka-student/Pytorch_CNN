import requests
import zipfile
import os

# @see http://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py

def download_file(url):
    """URL を指定してカレントディレクトリにファイルをダウンロードする
    """
    filename = '../data/tiny-imagenet-200.zip'
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
        return filename

    # ファイルが開けなかった場合は False を返す
    return False

def zip_extract(filename):
    """ファイル名を指定して zip ファイルをdataディレクトリに展開する
    """
    target_directory = '../data/'
    zfile = zipfile.ZipFile(filename)
    zfile.extractall(target_directory)


def data_download():
    if not os.path.exists('../data/tiny-imagenet-200.zip'):
        file_name = download_file("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    if not os.path.exists('../data/tiny-imagenet-200'):
        zip_extract('../data/tiny-imagenet-200.zip')

    return "tiny-imagenet-200"