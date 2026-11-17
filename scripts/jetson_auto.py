"""Git checkout veya USB'den yerel kurulum, CPU/varsa GPU ölçümü ve çıktı aktarımı."""
import argparse
import os, sys, json, time, shutil, hashlib, zipfile, platform, subprocess
import threading, traceback, shlex, signal, tarfile
from pathlib import Path
from datetime import datetime, timezone

KNOWN = {
    'ee864d9a01e4def14db77839b57bb2cb04709933d0d5c3668a9869c50d4f74f5': 2,
    '11c0b4bd167fb338a02e59924c6ac710e9eda10c37e1a1a31ca6dda8500cb1ae': 1,
}
BUNDLE_ZIP_SHA256 = '77e3c53def565ee50515de6910eb4a30b603119afcd823667ff3ef24eb0558f5'
REPO = None
SOURCE, PROJECT, REPORT = None, None, None
STATE = {'cpu': 'not_run', 'gpu': 'not_run', 'stage': 'başlangıç'}
TERMINAL = sys.stdout
LOCK = threading.RLock()

def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def textfile(path):
    try:
        # Some Jetson sysfs sensors return no data (None) temporarily. Passing
        # that through TextIO's decoder raises TypeError before OSError handling.
        data = Path(path).read_bytes()
        if data is None: return None
        return data.decode('utf-8', errors='replace').replace('\0', '').strip()
    except OSError: return None

def query(args, timeout=20):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, errors='replace', timeout=timeout)
        return {'code': p.returncode, 'stdout': p.stdout[-16000:], 'stderr': p.stderr[-8000:]}
    except Exception as exc: return {'code': -1, 'error': str(exc)}

def mount_id(path):
    found = query(['findmnt', '--json', '-T', str(path), '-o', 'SOURCE,TARGET,FSTYPE,UUID'])
    try: mount = json.loads(found.get('stdout', '{}'))['filesystems'][0]
    except (ValueError, KeyError, IndexError): mount = {}
    return {'device': path.stat().st_dev, 'mount': mount}

def same_usb(path, identity):
    try:
        current = mount_id(path)
        old_uuid = identity['mount'].get('uuid')
        if old_uuid: return current['mount'].get('uuid') == old_uuid
        return current == identity
    except OSError: return False

def walk(root, depth):
    if not root.is_dir(): return
    for folder, dirs, files in os.walk(str(root), followlinks=False):
        relative = Path(folder).relative_to(root)
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['c', 'd', 'wsl', 'node_modules', '__pycache__']]
        if len(relative.parts) >= depth: dirs[:] = []
        yield Path(folder), files

def package_manifest(path):
    try:
        if path.is_dir(): data = (path / 'PACKAGE_MANIFEST.json').read_bytes()
        else:
            with zipfile.ZipFile(str(path)) as z: data = z.read('PACKAGE_MANIFEST.json')
        key = digest(data)
        if key not in KNOWN: return None
        manifest = json.loads(data.decode('utf-8'))
        required = ['scripts/run_edge_benchmark.py', 'artifacts/research_bundle_portable/manifest.json']
        if not all(name in manifest['files'] for name in required): return None
        return (KNOWN[key], manifest)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile): return None

def copy_package(source, target, manifest):
    target.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(str(source)) if source.is_file() else None
    try:
        for name, expected in manifest['files'].items():
            dest = target / name
            if os.path.commonpath([str(target.resolve()), str(dest.resolve())]) != str(target.resolve()):
                raise RuntimeError('Paket dışına taşan dosya yolu: ' + name)
            data = archive.read(name) if archive else (source / name).read_bytes()
            if digest(data) != expected: raise RuntimeError('USB dosyası bozuk/değişmiş: ' + name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        raw = archive.read('PACKAGE_MANIFEST.json') if archive else (source/'PACKAGE_MANIFEST.json').read_bytes()
        (target/'PACKAGE_MANIFEST.json').write_bytes(raw)
    finally:
        if archive: archive.close()

def discover():
    global SOURCE
    choices = []
    for root in [Path('/media'), Path('/run/media'), Path('/mnt')]:
        for folder, _ in walk(root, 5):
            name = folder.name.casefold().replace('ı', 'i').replace(' ', '').replace('_', '').replace('-', '')
            if name != 'jetsonicinolan': continue
            if SOURCE is None:
                SOURCE = {'folder': str(folder.resolve()), 'identity': mount_id(folder)}
                save(REPORT/'usb_source.json', SOURCE)
            for inner, files in walk(folder, 6):
                candidates = ([inner] if 'PACKAGE_MANIFEST.json' in files else [])
                candidates += [inner/n for n in files if n.lower().endswith('.zip')]
                for candidate in candidates:
                    valid = package_manifest(candidate)
                    if valid: choices.append((valid[0], str(candidate), candidate, folder, valid[1]))
        if choices: break
    if not choices:
        raise RuntimeError('USB içindeki jetsonicinolan klasöründe verdiğim kaynak+model ZIP bulunamadı. '
                           'Dosya yöneticisinden USB’yi açıp yeniden dene; klasörün içinde tam hazırlık ZIP’i olmalı.')
    return sorted(choices, key=lambda item: (item[0], item[1]), reverse=True)[0]

def find_output_usb():
    """USB is optional in repository mode; never require a source package there."""
    folders = {}
    for root in [Path('/media'), Path('/run/media'), Path('/mnt')]:
        for folder, _ in walk(root, 5):
            name = folder.name.casefold().replace('ı', 'i').replace(' ', '').replace('_', '').replace('-', '')
            if name == 'jetsonicinolan': folders[str(folder.resolve())] = folder
    if len(folders) == 1:
        folder = next(iter(folders.values()))
        return {'folder': str(folder.resolve()), 'identity': mount_id(folder)}
    if folders:
        print('Birden fazla jetsonicinolan bulundu; otomatik USB hedefi seçilmedi:', list(folders))
    return None

def extract_repo_bundle(archive, target):
    if digest(archive.read_bytes()) != BUNDLE_ZIP_SHA256:
        raise RuntimeError('Repodaki model ZIP SHA-256 doğrulaması başarısız. Dosyayı yeniden indir.')
    with zipfile.ZipFile(str(archive)) as z:
        for name in z.namelist():
            if name == 'MODEL_PACKAGE_MANIFEST.json': continue
            dest = target/name
            if (not name.startswith('artifacts/research_bundle_portable/') or
                    os.path.commonpath([str(target.resolve()), str(dest.resolve())]) != str(target.resolve())):
                raise RuntimeError('Model ZIP içinde geçersiz yol: ' + name)
            if name.endswith('/'): continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
    bundle = target/'artifacts/research_bundle_portable'
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = bundle/name
        if (os.path.commonpath([str(bundle.resolve()), str(path.resolve())]) != str(bundle.resolve()) or
                digest(path.read_bytes()) != expected):
            raise RuntimeError('Model dosyası doğrulanamadı: ' + name)
    return manifest

def prepare_repo(repo, target):
    """Freeze only runtime sources; never modify the checkout or reuse its artifacts."""
    if not (repo/'scripts/run_edge_benchmark.py').is_file():
        raise RuntimeError('Jetson dalının kaynak dosyaları bulunamadı: ' + str(repo))
    archive = repo/'deploy/jetson/Research_Bundle_Functional_Check.zip'
    if not archive.is_file():
        raise RuntimeError('Hazır model ZIP bulunamadı. jetson-benchmark dalını güncelle.')
    record = {'kind': 'git_checkout', 'path': str(repo),
              'commit': query(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
              'status': query(['git', '-C', str(repo), 'status', '--porcelain']),
              'model_zip_sha256': BUNDLE_ZIP_SHA256}
    save(REPORT/'repository_source.json', record)
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    for folder in ['src', 'scripts', 'docs', 'tests']:
        entries += [p for p in (repo/folder).rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p.suffix != '.pyc']
    entries += [repo/name for name in ['README.md', 'requirements-edge.txt', 'requirements.txt',
                'One_Engine_Data.xlsx', 'Two_Engine_Data.xlsx', 'BASLAT.sh', 'JETSON_SAATI_DUZELT.sh',
                'USB_KONTROL.sh', 'SONUCLARI_GONDER.sh', 'GITHUB_GIRIS.sh', 'GPU_PYTORCH_KUR.sh']]
    for path in entries:
        if os.path.commonpath([str(repo.resolve()), str(path.resolve())]) != str(repo.resolve()):
            raise RuntimeError('Repo dışına çıkan kaynak yolu: ' + str(path))
        dest = target/path.relative_to(repo)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    bundle = extract_repo_bundle(archive, target)
    record['experiment_label'] = bundle['experiment_label']
    files = {p.relative_to(target).as_posix(): digest(p.read_bytes())
             for p in target.rglob('*') if p.is_file()}
    save(target/'PACKAGE_MANIFEST.json', {'source': record, 'files': files})
    save(REPORT/'repository_source.json', record)
    save(REPORT/'runtime_source_manifest.json', {'source': record, 'files': files})
    return bundle

def profile(base, filename='device_profile.json'):
    paths = []
    for candidate in [sys.executable, 'python3.12', 'python3.11', 'python3.10', 'python3']:
        found = shutil.which(candidate)
        if found and found not in paths: paths.append(found)
    signature = {'model': textfile('/proc/device-tree/model'), 'l4t': textfile('/etc/nv_tegra_release'),
                 'machine': platform.machine(), 'kernel': platform.release(), 'python_paths': paths}
    key = digest(json.dumps(signature, sort_keys=True).encode())
    cached = base/'device_profile.json'
    try: info = json.loads(cached.read_text())
    except (OSError, ValueError): info = {}
    reused = info.get('signature') == key
    if not reused:
        interpreters = []
        for executable in paths:
            result = query([executable, '-c', 'import sys,json; print(json.dumps(list(sys.version_info[:3])))'])
            try: version = json.loads(result['stdout'].strip())
            except (ValueError, KeyError): continue
            interpreters.append({'path': executable, 'version': version})
        info = {'signature': key, 'first_scan_utc': stamp(), 'identity': signature,
                'interpreters': interpreters, 'jetpack_package': query(['dpkg-query', '-W', 'nvidia-jetpack'])}
    info['cache_reused'] = reused
    info['checked_utc'] = stamp()
    info['current'] = {'ram': textfile('/proc/meminfo'), 'disk_free_bytes': shutil.disk_usage(str(base)).free,
                       'cpu_count': os.cpu_count(), 'nvpmodel': query(['nvpmodel', '-q']),
                       'clocks': query(['jetson_clocks', '--show']), 'cpu': query(['lscpu']),
                       'thermal': {textfile(p/'type') or p.name: textfile(p/'temp')
                                   for p in Path('/sys/class/thermal').glob('thermal_zone*')},
                       'cuda_toolkit': query([shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc', '--version'])}
    save(cached, info)
    save(REPORT/filename, info)
    return info

def phase(name):
    STATE['stage'] = name
    print('\n=== ' + name + ' ===', flush=True)

def run(args, timeout=1800, env=None, optional=False):
    print('$ ' + ' '.join(shlex.quote(str(x)) for x in args), flush=True)
    detached = str(args[0]) != 'sudo'
    p = subprocess.Popen([str(x) for x in args], cwd=str(PROJECT), env=env,
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True, errors='replace', start_new_session=detached)
    def pump():
        for line in p.stdout:
            with LOCK: print(line, end='', flush=True)
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    try: p.wait(timeout=timeout)
    except BaseException:
        try:
            if detached: os.killpg(p.pid, signal.SIGTERM)
            else: p.terminate()
        except ProcessLookupError: pass
        try: p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if detached: os.killpg(p.pid, signal.SIGKILL)
            else: p.kill()
            p.wait()
        raise
    finally: reader.join(timeout=10)
    if p.returncode and not optional: raise RuntimeError('Komut başarısız; çıkış kodu ' + str(p.returncode))
    return p.returncode == 0

def apt(packages):
    if os.geteuid() == 0: prefix = []
    else:
        if not shutil.which('sudo'): raise RuntimeError('Eksik sistem paketi var, sudo yok: ' + ', '.join(packages))
        with open('/dev/tty', 'rb', buffering=0) as tty_in, open('/dev/tty', 'wb', buffering=0) as tty_out:
            if subprocess.call(['sudo', '-v'], stdin=tty_in, stdout=tty_out, stderr=tty_out):
                raise RuntimeError('Sudo yetkisi alınamadı; ihtiyaç: ' + ', '.join(packages))
        prefix = ['sudo', '-n']
    run(prefix + ['apt-get', '-o', 'Acquire::Retries=2', '-o', 'Acquire::http::Timeout=25', 'update'], timeout=360)
    run(prefix + ['apt-get', '-y', 'install'] + packages, timeout=600)

def make_env(base, executable, version, mode):
    folder = base/('venv_' + mode + '_' + ''.join(str(x) for x in version[:2]))
    py = folder/'bin/python'
    if py.exists() and query([str(py), '-m', 'pip', '--version'])['code'] == 0: return py
    if folder.exists(): folder = base/(folder.name + '_' + stamp())
    args = [executable, '-m', 'venv']
    if mode == 'gpu': args.append('--system-site-packages')
    if not run(args + [str(folder)], optional=True):
        apt(['python{}.{}-venv'.format(*version[:2])])
        folder = base/('venv_' + mode + '_' + stamp())
        run(args + [str(folder)])
    return folder/'bin/python'

def torch_probe(executable):
    code = ("import torch,json; ok=torch.cuda.is_available(); "
            "x=torch.ones(1,device='cuda') if ok else None; "
            "torch.cuda.synchronize() if ok else None; "
            "print(json.dumps({'version':torch.__version__,'cuda':ok,'build':torch.version.cuda,"
            "'gpu':torch.cuda.get_device_name(0) if ok else None}))")
    result = query([str(executable), '-c', code], timeout=90)
    try: return json.loads(result.get('stdout', '').strip().splitlines()[-1])
    except (ValueError, IndexError): return {'cuda': False, 'error': result}

def registered_gpu_runtime(base, identity):
    """Accept only a verified private runtime registered for this Jetson identity."""
    try:
        record = json.loads((base/'gpu_runtime.json').read_text(encoding='utf-8'))
        path = Path(record['python'])
        matches = all(record['identity'].get(key) == identity.get(key) for key in ['machine', 'model', 'l4t'])
        if (record.get('schema_version') != 1 or not record.get('probe', {}).get('cuda') or
                not matches or not path.is_file() or
                os.path.commonpath([str(base.resolve()), str(path.resolve())]) != str(base.resolve())):
            return None
        version = record['version']
        if not (3,10) <= tuple(version[:2]) <= (3,12): return None
        return {'path': str(path), 'version': version, 'registered_runtime': True}
    except (OSError, ValueError, KeyError, TypeError):
        return None

def benchmark(py, mode, short):
    output = REPORT/(mode + ('_smoke' if short else '_benchmark'))
    args = [str(py), 'scripts/run_edge_benchmark.py', '--bundle', str(PROJECT/'artifacts/research_bundle_portable'),
            '--ft-device', 'cuda' if mode == 'gpu' else 'cpu', '--sample-size', '24' if short else '200',
            '--warmup', '5' if short else '50', '--repetitions', '20' if short else '1000',
            '--trials', '1' if short else '3', '--threads', '1', '--seed', '42',
            '--require-jetson', '--output', str(output)]
    success = run(args, timeout=7200, optional=True)
    try: report = json.loads((output/'edge_benchmark_latest.json').read_text())
    except (OSError, ValueError): return False
    return success and report.get('complete') is True and report.get('verified_jetson') is True

def main(base):
    global SOURCE, PROJECT
    phase('Cihaz profili: önbellek kontrolü ve güncel durum')
    info = profile(base)
    print(json.dumps(info['identity'], ensure_ascii=False), flush=True)
    print('Cihaz profili önbellekten:', info['cache_reused'], flush=True)
    if shutil.disk_usage(str(base)).free < 3*1024**3: raise RuntimeError('Yerel diskte en az 3 GiB boş alan gerekiyor.')
    PROJECT = REPORT.parent/'project'
    if REPO is not None:
        phase('Repodan kaynak kopyası ve hazır model bütünlük kontrolü')
        SOURCE = find_output_usb()
        if SOURCE: save(REPORT/'usb_source.json', SOURCE)
        manifest = prepare_repo(REPO, PROJECT)
        print('Hazır model etiketi:', manifest['experiment_label'], '/ FT kısa eğitim: 5 epoch', flush=True)
    else:
        phase('USB paketini bulma ve Jetson diskine doğrulanmış kopya')
        _, _, package, usb, manifest = discover()
        SOURCE = {'folder': str(usb.resolve()), 'identity': mount_id(usb)}
        save(REPORT/'usb_source.json', SOURCE)
        copy_package(package, PROJECT, manifest)
    os.chdir(str(PROJECT))
    print('KOPYA TAMAM. Çalışma Jetson diskindeki doğrulanmış kopyadan devam ediyor.', flush=True)
    if SOURCE:
        print('USB’yi güvenle çıkarabilirsin. Takılı bırakırsan sonuçlar bitince USB’ye de kopyalanır.', flush=True)
    else:
        print('USB gerekmiyor; sonuçlar Jetson diskine kaydedilecek.', flush=True)
    if info['identity']['machine'] != 'aarch64' or not info['identity']['l4t'] or not info['identity']['model']:
        raise RuntimeError('Bu ortamda gerçek Jetson kimliği doğrulanamadı; cihaz kaydı kaydedildi.')
    available = [p for p in info['interpreters'] if (3,10) <= tuple(p['version'][:2]) <= (3,12)]
    if not available: raise RuntimeError('Python 3.10–3.12 bulunamadı. JetPack/Python uyarlaması gerekiyor; cihaz çıktısını gönder.')
    chosen = available[0]
    if shutil.disk_usage(str(base)).free < 3*1024**3: raise RuntimeError('Yerel diskte en az 3 GiB boş alan gerekiyor.')
    phase('Gerekli sistem kütüphanesi ve izole CPU ortamı')
    if query([chosen['path'], '-c', "import ctypes; ctypes.CDLL('libgomp.so.1')"])['code']:
        apt(['libgomp1'])
    # Eski 12 Ekim exportu Git metadata kaydı için git çalıştırır.
    if not shutil.which('git') and 'except OSError:' not in (PROJECT/'src/evaluation/research_bundle.py').read_text():
        apt(['git'])
    cpu = make_env(base, chosen['path'], chosen['version'], 'cpu')
    pip = [str(cpu), '-m', 'pip', '--disable-pip-version-check', '--no-input', 'install',
           '--only-binary=:all:', '--retries', '2', '--timeout', '25',
           '--cache-dir', str(base/'pip-cache'), '--index-url', 'https://pypi.org/simple']
    phase('CPU bağımlılıkları; internet veya pip önbelleği kullanılır')
    run(pip + ['-r', 'requirements-edge.txt'])
    run(pip + ['torch==2.5.1'])
    run([str(cpu), '-m', 'pip', 'check'])
    save(REPORT/'cpu_packages.json', query([str(cpu), '-m', 'pip', 'freeze']))
    run([str(cpu), '-c', "from pathlib import Path; from src.evaluation.research_bundle import verify_bundle; m=verify_bundle(Path('artifacts/research_bundle_portable')); print('Model etiketi:',m['experiment_label'])"])
    save(REPORT/'cpu_torch.json', torch_probe(cpu))
    phase('Kısa CPU denemesi')
    if not benchmark(cpu, 'cpu', True):
        STATE['cpu'] = 'smoke_failed'
        raise RuntimeError('Kısa CPU denemesi başarısız; loglar ve cihaz profili korunuyor.')
    phase('Üç yöntem CPU benchmarkı: 3 tekrar, 1000 çağrı')
    STATE['cpu'] = 'passed' if benchmark(cpu, 'cpu', False) else 'benchmark_failed'
    phase('Mevcut NVIDIA CUDA ortamı kontrolü')
    gpu_interpreter = chosen
    system_torch = {'cuda': False}
    system_probes = []
    registered = registered_gpu_runtime(base, info['identity'])
    gpu_candidates = ([registered] if registered else []) + available
    for candidate in gpu_candidates:
        probe = torch_probe(candidate['path'])
        system_probes.append({'interpreter': candidate, 'probe': probe})
        if probe.get('cuda'):
            gpu_interpreter, system_torch = candidate, probe
            break
    save(REPORT/'system_torch_candidates.json', system_probes)
    save(REPORT/'system_torch.json', system_torch)
    STATE['gpu'] = 'skipped_cuda_not_ready'
    if system_torch.get('cuda'):
        try:
            # A venv does not inherit another venv's site-packages. Use the
            # registered CUDA runtime directly so its NVIDIA torch is retained.
            gpu = (Path(gpu_interpreter['path']) if gpu_interpreter.get('registered_runtime') else
                   make_env(base, gpu_interpreter['path'], gpu_interpreter['version'], 'gpu'))
            gpip = [str(gpu)] + pip[1:]
            run(gpip + ['-r', 'requirements-edge.txt'])
            run([str(gpu), '-m', 'pip', 'check'])
            save(REPORT/'gpu_packages.json', query([str(gpu), '-m', 'pip', 'freeze']))
            gp = torch_probe(gpu)
            save(REPORT/'gpu_torch.json', gp)
            if gp.get('cuda') and benchmark(gpu, 'gpu', True):
                STATE['gpu'] = 'passed' if benchmark(gpu, 'gpu', False) else 'benchmark_failed'
            else: STATE['gpu'] = 'cuda_or_smoke_failed'
        except Exception as exc:
            STATE['gpu'] = 'setup_failed'
            STATE['gpu_error'] = str(exc)
            traceback.print_exc()
    else:
        print('GPU ölçümü atlandı: kayıtlı veya sistem PyTorch/CUDA hazır değil. Rastgele NVIDIA wheel/driver kurulmadı.', flush=True)
    phase('Son cihaz durumu ve arayüz başlatıcısı')
    profile(base, 'device_after.json')
    launcher = REPORT.parent/'ARAYUZU_AC.sh'
    launch = ('#!/usr/bin/env bash\ncd ' + shlex.quote(str(PROJECT)) + '\n'
        'export SPECIFIC_RANGE_BUNDLE=' + shlex.quote(str(PROJECT/'artifacts/research_bundle_portable')) + '\n'
        'export SPECIFIC_RANGE_EDGE_RESULTS=' + shlex.quote(str(REPORT/'cpu_benchmark/edge_benchmark_latest.json')) + '\n'
        'export SPECIFIC_RANGE_NO_BROWSER=1\nexec ' + shlex.quote(str(cpu)) + ' scripts/web_app/server.py\n')
    launcher.write_text(launch, encoding='utf-8')
    launcher.chmod(0o700)
    STATE['ui_launcher'] = str(launcher)
    STATE['stage'] = 'tamamlandı'

def deliver(base, log):
    STATE['finished_utc'] = stamp()
    save(REPORT/'STATUS.json', STATE)
    (REPORT/'OKU_BENI.txt').write_text(
        'Yerel sonuçlar: ' + str(REPORT) + '\nCPU: ' + STATE['cpu'] + '\nGPU: ' + STATE['gpu'] +
        '\nPaket modelleri functional_check / kısa eğitimdir; nihai bilimsel model değildir.\n'
        'Cihaz profili, kurulum logu, STATUS ve varsa bütün benchmark ham dosyaları buradadır.\n', encoding='utf-8')
    log.flush()
    sys.stdout = TERMINAL
    sys.stderr = TERMINAL
    log.close()
    hashes = {p.relative_to(REPORT).as_posix(): digest(p.read_bytes()) for p in REPORT.rglob('*') if p.is_file()}
    save(REPORT/'OUTPUT_SHA256.json', hashes)
    archive = REPORT.parent/(REPORT.parent.name + '_sonuclar.tar.gz')
    with tarfile.open(str(archive), 'w:gz') as tar: tar.add(str(REPORT), arcname='output')
    print('\nCPU:', STATE['cpu'], '| GPU:', STATE['gpu'])
    print('YEREL ÇIKTI:', REPORT)
    print('BANA GÖNDERECEĞİN ARŞİV:', archive)
    if STATE.get('ui_launcher'): print('Arayüz için:', 'bash ' + shlex.quote(STATE['ui_launcher']), ' -> http://localhost:5000')
    if SOURCE and same_usb(Path(SOURCE['folder']), SOURCE['identity']):
        try:
            dest = Path(SOURCE['folder'])/'output'/REPORT.parent.name
            if dest.exists(): raise RuntimeError('USB hedefi zaten var; üzerine yazılmadı.')
            if shutil.disk_usage(str(Path(SOURCE['folder']))).free < archive.stat().st_size + sum(p.stat().st_size for p in REPORT.rglob('*') if p.is_file()) + 10*1024**2:
                raise RuntimeError('USB boş alanı yetersiz.')
            shutil.copytree(str(REPORT), str(dest))
            shutil.copy2(str(archive), str(dest/archive.name))
            for name, expected in hashes.items():
                if digest((dest/name).read_bytes()) != expected: raise RuntimeError('USB doğrulaması başarısız: ' + name)
            if digest((dest/archive.name).read_bytes()) != digest(archive.read_bytes()): raise RuntimeError('USB arşiv doğrulaması başarısız.')
            print('USB ÇIKTISI KOPYALANDI VE DOĞRULANDI:', dest)
        except Exception as exc: print('USB’ye aktarım tamamlanamadı; yerel kopyalar korunuyor:', exc)
    else: print('Uygun USB hedefi yok veya çıkarılmış. Çıktılar Jetson diskinde korundu.')
    print('Herhangi bir hata olduysa STATUS.json ve kurulum.log da arşivin içindedir.')

def entry():
    global REPORT, REPO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Klonlanan Jetson repo klasörü (varsayılan: bu betiğin reposu)')
    parser.add_argument('--usb', action='store_true', help='Eski kaynak+model USB paketinden çalıştır')
    parser.add_argument('--no-publish', action='store_true', help='GitHub sonuç gönderimini deneme')
    args = parser.parse_args()
    REPO = None if args.usb else args.repo.resolve()
    if os.name != 'posix': raise RuntimeError('Bu komut Jetson Linux terminali içindir.')
    base = Path.home()/'specific-range-jetson-auto'
    REPORT = base/('run_' + stamp())/'output'
    REPORT.mkdir(parents=True, exist_ok=False)
    log = (REPORT/'kurulum.log').open('a', encoding='utf-8', buffering=1)
    class Tee:
        def write(self, value):
            with LOCK: TERMINAL.write(value); log.write(value)
            return len(value)
        def flush(self):
            with LOCK: TERMINAL.flush(); log.flush()
    sys.stdout = sys.stderr = Tee()
    os.environ.update({'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('PYTHONHOME', None)
    try: main(base)
    except BaseException as exc:
        STATE['error'] = str(exc) or type(exc).__name__
        traceback.print_exc()
        print('\nDURDU: ' + STATE['error'], flush=True)
    finally:
        try: deliver(base, log)
        except BaseException:
            sys.stdout = sys.stderr = TERMINAL
            traceback.print_exc()
            print('Yerel kayıtlar:', REPORT)
    if not args.no_publish and REPO is not None and STATE.get('error') != 'KeyboardInterrupt':
        try:
            subprocess.call([sys.executable, str(REPO/'scripts/publish_jetson_results.py'),
                             '--repo', str(REPO), '--output', str(REPORT)], timeout=240)
        except Exception as exc:
            print('GitHub gönderimi tamamlanamadı; yerel sonuçlar korunuyor:', exc)
    if STATE.get('error') or STATE['cpu'] != 'passed':
        sys.exit(130 if STATE.get('error') == 'KeyboardInterrupt' else 1)

if __name__ == '__main__': entry()
