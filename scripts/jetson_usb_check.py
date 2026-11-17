"""Diagnose USB size/partitions and try a read-only exFAT mount; never repair/format."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

SYSTEM_MOUNTS = {'/', '/boot', '/boot/efi', '/home', '/usr', '/var', '[SWAP]'}
SUDO_AUTH = {}


def query(args, timeout=25):
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, errors='replace', timeout=timeout)
        return {'args': args, 'code': result.returncode,
                'stdout': result.stdout[-50000:], 'stderr': result.stderr[-12000:]}
    except Exception as exc:
        return {'args': args, 'code': -1, 'error': str(exc)}


def nodes(tree):
    yield tree
    for child in tree.get('children', []):
        for node in nodes(child): yield node


def mountpoints(node):
    value = node.get('mountpoints') or node.get('mountpoint') or []
    return [value] if isinstance(value, str) else [p for p in value if p]


def usb_disks(payload):
    found = []
    for disk in payload.get('blockdevices', []):
        if disk.get('type') != 'disk': continue
        removable = str(disk.get('rm')).lower() in ('true', '1')
        if disk.get('tran') != 'usb' and not (removable and disk.get('name', '').startswith('/dev/sd')):
            continue
        if any(SYSTEM_MOUNTS.intersection(mountpoints(n)) for n in nodes(disk)):
            continue
        found.append(disk)
    return found


def exfat_candidates(disks):
    found = []
    for disk in disks:
        for node in nodes(disk):
            if (node.get('fstype') or '').lower() == 'exfat' and int(node.get('size') or 0) >= 1024**2:
                found.append(node)
    return found


def read_layout():
    result = query(['lsblk', '--json', '--bytes', '--paths', '-o',
                    'NAME,TYPE,SIZE,FSTYPE,LABEL,MOUNTPOINT,TRAN,RM,RO'])
    if result['code']: raise RuntimeError('lsblk okunamadı: ' + str(result))
    return json.loads(result['stdout']), result


def sudo_ready():
    global SUDO_AUTH
    if os.geteuid() == 0: return True
    if not shutil.which('sudo'): return False
    try:
        # A terminal is non-seekable: buffered r+ fails before sudo can prompt.
        with open('/dev/tty', 'rb', buffering=0) as tty_in, open('/dev/tty', 'wb', buffering=0) as tty_out:
            code = subprocess.call(['sudo', '-v'], stdin=tty_in, stdout=tty_out, stderr=tty_out)
            SUDO_AUTH = {'method': 'controlling_terminal', 'code': code}
            return code == 0
    except OSError as exc:
        result = query(['sudo', '-n', 'true'])
        SUDO_AUTH = {'method': 'cached_only', 'terminal_error': str(exc), 'result': result}
        return result['code'] == 0


def try_mount(candidate, privileged):
    if mountpoints(candidate): return {'status': 'already_mounted', 'paths': mountpoints(candidate)}
    device = candidate['name']
    attempts = []
    if shutil.which('udisksctl'):
        result = query(['udisksctl', 'mount', '--block-device', device,
                        '--options', 'ro', '--no-user-interaction'])
        attempts.append(result)
        if result['code'] == 0: return {'status': 'mounted_read_only', 'attempts': attempts}
    fuse = shutil.which('mount.exfat-fuse') or (
        '/sbin/mount.exfat-fuse' if Path('/sbin/mount.exfat-fuse').is_file() else None)
    if fuse and privileged:
        # Recheck identity and root exclusion immediately before touching a mount.
        layout, _ = read_layout()
        fresh = [n for n in exfat_candidates(usb_disks(layout)) if n['name'] == device
                 and n.get('size') == candidate.get('size') and n.get('label') == candidate.get('label')]
        if len(fresh) != 1: raise RuntimeError('USB aygıtı değişti; bağlama iptal edildi')
        if mountpoints(fresh[0]): return {'status': 'already_mounted', 'paths': mountpoints(fresh[0])}
        parent = Path('/media')/getpass.getuser()
        target = parent/('jetson-usb-' + Path(device).name)
        if target.is_symlink() or parent.is_symlink(): raise RuntimeError('Sembolik bağlama yolu kullanılmadı')
        if target.exists() and any(target.iterdir()): raise RuntimeError('Dolu bağlama klasörü kullanılmadı')
        prefix = [] if os.geteuid() == 0 else ['sudo', '-n']
        made = query(prefix + ['mkdir', '-p', '--', str(target)])
        attempts.append(made)
        if made['code'] == 0:
            result = query(prefix + [fuse, '-o', 'ro,uid={},gid={}'.format(os.getuid(), os.getgid()),
                                     device, str(target)])
            attempts.append(result)
            if result['code'] == 0:
                return {'status': 'mounted_read_only', 'paths': [str(target)], 'attempts': attempts}
    return {'status': 'mount_failed', 'attempts': attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-publish', action='store_true', help='Yalnız yerel rapor; GitHub gönderimi deneme')
    args = parser.parse_args()
    if os.name != 'posix': raise RuntimeError('Bu araç Jetson Linux içindir')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = Path.home()/'specific-range-jetson-auto'/('usb_' + stamp)
    report = folder/'output'
    report.mkdir(parents=True, exist_ok=False)
    info = {'status': 'collecting', 'created_utc': stamp, 'read_only_mount_requested': True}
    try:
        privileged = sudo_ready()
        prefix = [] if os.geteuid() == 0 else ['sudo', '-n']
        info['sudo_ready'] = privileged
        info['sudo_auth'] = SUDO_AUTH
        if not privileged:
            print('YONETICI YETKISI ALINAMADI. FUSE baglama ve ayrintili disk sorgulari yetki bekliyor.')
        commands = [(['uname', '-a'], 'kernel'), (['lsusb'], 'lsusb'),
                    (['lsusb', '-t'], 'usb_tree'), (['df', '-hT'], 'df'),
                    (['dpkg-query', '-W', 'exfat-fuse', 'exfat-utils', 'exfatprogs'], 'exfat_packages'),
                    (['findmnt', '--json'], 'mounts'),
                    (prefix + ['dmesg', '--ctime'], 'dmesg')]
        for command, name in commands:
            info[name] = query(command)
        info['os_release'] = Path('/etc/os-release').read_text(errors='replace')
        info['python'] = sys.version
        for path, key in [('/proc/device-tree/model', 'jetson_model'), ('/etc/nv_tegra_release', 'l4t')]:
            if Path(path).is_file(): info[key] = Path(path).read_text(errors='replace').replace('\0', '')
        info['filesystems'] = Path('/proc/filesystems').read_text(errors='replace')
        layout, raw = read_layout()
        info['lsblk_before'] = raw
        disks = usb_disks(layout)
        info['usb_disks'] = disks
        info['small_capacity_disks'] = [d['name'] for d in disks if int(d.get('size') or 0) < 1024**2]
        for disk in disks:
            name = disk['name']
            print('USB DISK:', name, '| BAYT:', disk.get('size'), '| ETIKET:', disk.get('label'))
            for node in nodes(disk):
                print('  ', node.get('name'), 'boyut=', node.get('size'),
                      'dosya_sistemi=', node.get('fstype'), 'bagli=', mountpoints(node))
            info.setdefault('partition_tables', []).append(query(prefix + ['fdisk', '-l', name]))
            info.setdefault('filesystem_probes', []).extend(query(prefix + ['blkid', '-p', n['name']]) for n in nodes(disk))
        if info['small_capacity_disks']:
            print('DISKIN TAMAMI 1 MiB altinda gorunuyor:', info['small_capacity_disks'])
            print('USB yi hub yerine dogrudan baska porta takarak tekrar dene; bu tek basina exFAT destegi hatasi degildir.')
        candidates = exfat_candidates(disks)
        if len(candidates) == 1:
            info['mount_attempt'] = try_mount(candidates[0], privileged)
            info['status'] = info['mount_attempt']['status']
        elif len(candidates) > 1:
            info['status'] = 'multiple_exfat_candidates'
            print('Birden fazla exFAT bölüm var; yanlış aygıt seçmemek için bağlama yapılmadı.')
        else:
            info['status'] = 'no_exfat_partition_detected'
            print('USB üzerinde tanınan büyük exFAT bölüm bulunamadı. 32 KB bölümü veri bölümü varsaymadım.')
        after, info['lsblk_after'] = read_layout()
        info['mount_paths'] = [p for d in usb_disks(after) for n in nodes(d) for p in mountpoints(n)]
        info['dmesg_after'] = query(prefix + ['dmesg', '--ctime'])
        for path in info['mount_paths']:
            print('USB ERISIM YOLU:', path)
        if info['status'] == 'mount_failed':
            for attempt in info['mount_attempt']['attempts']:
                print('BAGLAMA MESAJI:', attempt.get('stderr') or attempt.get('stdout') or attempt.get('error'))
    except BaseException as exc:
        info['status'], info['error'] = 'diagnostic_error', str(exc)
        print('USB KONTROL DURDU:', exc)
    finally:
        (report/'usb_report.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        text = ('USB durumu: ' + info['status'] + '\nBiçimlendirme veya dosya sistemi onarımı yapılmadı.\n'
                'Yeni bağlama denemeleri salt okunurdur.\n'
                'Tüm disk küçük mü, yalnız bir bölüm mü küçük: usb_report.json içindeki usb_disks/lsblk_before.\n')
        (report/'OKU_BENI.txt').write_text(text, encoding='utf-8')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in report.iterdir() if p.is_file()}
        (report/'OUTPUT_SHA256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
        archive = folder/(folder.name + '_sonuclar.tar.gz')
        with tarfile.open(str(archive), 'w:gz') as tar: tar.add(str(report), arcname='output')
        print('USB DURUMU:', info['status'])
        print('USB RAPORU:', archive)
    if not args.no_publish:
        try:
            subprocess.call([sys.executable, str(args.repo/'scripts/publish_jetson_results.py'),
                             '--repo', str(args.repo), '--output', str(report)], timeout=240)
        except Exception as exc: print('GitHub gönderimi yapılamadı; rapor yerelde:', exc)
    return 0 if info['status'] in ('already_mounted', 'mounted_read_only') else 1


if __name__ == '__main__': sys.exit(main())
