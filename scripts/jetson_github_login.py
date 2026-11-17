"""Install a user-local official GitHub CLI if missing and open browser login."""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request


def download(url, limit=80*1024**2):
    request = urllib.request.Request(url, headers={'User-Agent': 'specific-range-jetson'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(limit+1)
    if len(data) > limit: raise RuntimeError('İndirme boyut sınırını aştı')
    return data


def main():
    try:
        tools = Path.home()/'.local/share/specific-range-tools'
        os.environ['PATH'] = str(tools) + os.pathsep + os.environ.get('PATH', '')
        gh = shutil.which('gh')
        if not gh:
            arch = {'aarch64': 'arm64', 'x86_64': 'amd64'}.get(platform.machine())
            if not arch: raise RuntimeError('GitHub CLI mimarisi desteklenmedi: ' + platform.machine())
            print('Resmî GitHub CLI kullanıcı klasörüne indiriliyor; sistem paketi kurulmayacak.', flush=True)
            release = json.loads(download('https://api.github.com/repos/cli/cli/releases/latest').decode())
            assets = {item['name']: item['browser_download_url'] for item in release['assets']}
            archive_name = next(name for name in assets if name.endswith('_linux_' + arch + '.tar.gz'))
            checksums_name = next(name for name in assets if name.endswith('_checksums.txt'))
            checksums = download(assets[checksums_name]).decode()
            expected = next(line.split()[0] for line in checksums.splitlines()
                            if line.split()[-1].lstrip('*') == archive_name)
            data = download(assets[archive_name])
            if hashlib.sha256(data).hexdigest() != expected: raise RuntimeError('GitHub CLI SHA-256 eşleşmedi')
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
                member = archive.getmember(archive_name[:-7] + '/bin/gh')
                if not member.isfile(): raise RuntimeError('GitHub CLI dosyası geçersiz')
                binary = archive.extractfile(member).read()
            tools.mkdir(parents=True, exist_ok=True)
            target = tools/'gh'
            target.write_bytes(binary)
            target.chmod(0o700)
            gh = str(target)
        print('GitHub girişini tarayıcıda tamamla. Şifre veya tokenı bu sohbete gönderme.', flush=True)
        subprocess.check_call([gh, 'auth', 'login', '--hostname', 'github.com', '--git-protocol', 'https', '--web'])
        subprocess.check_call([gh, 'auth', 'setup-git', '--hostname', 'github.com'])
        print('GİRİŞ TAMAM. Şimdi: bash SONUCLARI_GONDER.sh')
        return 0
    except Exception as exc:
        print('GitHub girişi tamamlanamadı:', exc)
        print('HTTPS/tarih hatası varsa önce: bash JETSON_SAATI_DUZELT.sh')
        return 1


if __name__ == '__main__': sys.exit(main())
