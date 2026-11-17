"""Publish collected Jetson output, never the user's checkout or credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

REMOTE = 'https://github.com/ufukzkn/specific-range-studio.git'
BRANCH = 'jetson-benchmark'


def git(args, cwd, env, check=True):
    result = subprocess.run(['git'] + list(args), cwd=str(cwd), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, errors='replace', timeout=90)
    if check and result.returncode:
        raise RuntimeError('Git ' + args[0] + ': ' + (result.stderr or result.stdout)[-2500:])
    return result


def collect_files(report):
    files = []
    for path in sorted(report.rglob('*')):
        if path.is_symlink(): raise RuntimeError('Sembolik çıktı yolu gönderilmedi: ' + str(path))
        if not path.is_file(): continue
        if '.git' in path.relative_to(report).parts: raise RuntimeError('Çıktıda .git bulundu')
        if path.stat().st_size > 45*1024**2: raise RuntimeError('Git için büyük çıktı dosyası: ' + str(path))
        files.append(path)
    if not files: raise RuntimeError('Çıktı klasörü boş: ' + str(report))
    return files


def publish(repo, reports, base, remote=REMOTE):
    env = os.environ.copy()
    env.update({'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'never', 'GH_PROMPT_DISABLED': '1'})
    tools = Path.home()/'.local/share/specific-range-tools'
    env['PATH'] = str(tools) + os.pathsep + env.get('PATH', '')
    gh = shutil.which('gh', path=env['PATH'])
    if gh:
        # Command-scoped helper; no token is printed or written into a repository.
        env['GIT_CONFIG_COUNT'] = '2'
        env['GIT_CONFIG_KEY_0'], env['GIT_CONFIG_VALUE_0'] = 'credential.helper', ''
        env['GIT_CONFIG_KEY_1'], env['GIT_CONFIG_VALUE_1'] = 'credential.helper', '!gh auth git-credential'
    workspace = base/'git-publish'/('upload_' + uuid.uuid4().hex)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    print('GITHUB: Çıktılar ayrı yerel Git kopyasında hazırlanıyor...', flush=True)
    git(['clone', '--depth', '1', '--branch', BRANCH, '--single-branch', remote, str(workspace)], base, env)
    # Preserve the Jetson checkout and never commit local source/model modifications.
    hashes = {}
    for report in reports:
        prefix = report.parent.name
        if not (prefix.startswith('run_') or prefix.startswith('usb_')):
            raise RuntimeError('Bilinmeyen çıktı klasörü: ' + str(report))
        destination = workspace/'jetson-results'/prefix
        for path in collect_files(report):
            relative = path.relative_to(report)
            dest = destination/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            dest.write_bytes(data)
            hashes[(Path('jetson-results')/prefix/relative).as_posix()] = hashlib.sha256(data).hexdigest()
    name = git(['config', 'user.name'], repo, env, check=False).stdout.strip() or 'Jetson benchmark'
    email = git(['config', 'user.email'], repo, env, check=False).stdout.strip() or 'jetson@localhost'
    git(['config', 'user.name', name], workspace, env)
    git(['config', 'user.email', email], workspace, env)
    git(['config', 'commit.gpgsign', 'false'], workspace, env)
    hooks = workspace/'empty-hooks'
    hooks.mkdir()
    git(['config', 'core.hooksPath', str(hooks)], workspace, env)
    git(['add', '--', 'jetson-results'], workspace, env)
    changed = git(['diff', '--cached', '--quiet'], workspace, env, check=False)
    if changed.returncode not in (0, 1): raise RuntimeError('Git değişiklik kontrolü başarısız')
    if changed.returncode == 1:
        git(['commit', '-m', 'Record Jetson output: ' + ', '.join(p.parent.name for p in reports)], workspace, env)
    receipt = {'status': 'committed_locally', 'branch': BRANCH, 'workspace': str(workspace),
               'commit': git(['rev-parse', 'HEAD'], workspace, env).stdout.strip(), 'files': hashes}
    try:
        print("GITHUB: Sonuç commit'i jetson-benchmark dalına gönderiliyor...", flush=True)
        git(['push', 'origin', 'HEAD:refs/heads/' + BRANCH], workspace, env)
        receipt['status'] = 'pushed'
        receipt['url'] = 'https://github.com/ufukzkn/specific-range-studio/tree/' + BRANCH + '/jetson-results'
    except Exception as exc:
        receipt['error'] = str(exc)
    for report in reports:
        (report.parent/'github_upload.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Tek run_.../output veya usb_.../output; varsayılan tüm yerel koşular')
    args = parser.parse_args()
    base = Path.home()/'specific-range-jetson-auto'
    validated = []
    try:
        if not shutil.which('git'): raise RuntimeError('Git kurulu değil')
        reports = [args.output.resolve()] if args.output else sorted(
            list(base.glob('run_*/output')) + list(base.glob('usb_*/output')))
        if not reports: raise RuntimeError('Gönderilecek cihaz çıktısı bulunamadı')
        for report in reports:
            if not report.is_dir() or os.path.commonpath([str(base.resolve()), str(report.resolve())]) != str(base.resolve()):
                raise RuntimeError('Yerel çalışma alanı dışında çıktı gönderilmedi: ' + str(report))
            validated.append(report)
        receipt = publish(args.repo.resolve(), reports, base)
        if receipt['status'] != 'pushed': raise RuntimeError(receipt['error'])
        print('GITHUB SONUÇLARI GÖNDERİLDİ:', receipt['url'])
        print('Commit:', receipt['commit'])
        return 0
    except Exception as exc:
        for report in validated:
            receipt = report.parent/'github_upload.json'
            if not receipt.exists():
                receipt.write_text(json.dumps({'status': 'upload_failed', 'error': str(exc),
                                               'branch': BRANCH}, indent=2), encoding='utf-8')
        print('GITHUB GÖNDERİMİ TAMAMLANAMADI:', exc)
        print('Yerel çıktılar korunuyor. İnternet ve yazma yetkisi gerekir.')
        print('Repo klasöründe: bash GITHUB_GIRIS.sh')
        print('Ardından: bash SONUCLARI_GONDER.sh')
        return 1


if __name__ == '__main__': sys.exit(main())
