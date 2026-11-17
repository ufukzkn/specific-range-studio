"""Install/test an isolated NVIDIA CUDA PyTorch runtime; never run benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request
import zipfile

import jetson_auto as auto

TORCH_URL = ('https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/'
             'torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl')
# Canonical wheel filename follows its internal METADATA version; bytes stay unchanged.
TORCH_FILE = 'torch-2.5.0a0+872d972e41.nv24.8-cp310-cp310-linux_aarch64.whl'
TORCH_BYTES = 806950107
SPARSE_URL = ('https://developer.download.nvidia.com/compute/cusparselt/redist/'
              'libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz')
SPARSE_BYTES = 23133384
SPARSE_SHA256 = '389167df60404fe9d49f5975239bd095c440328d951152082d8603f8bdac1a97'
SPARSE_PREFIX = 'libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/'
LIB_PACKAGES = {
    'libopenblas.so.0': 'libopenblas0', 'libgomp.so.1': 'libgomp1',
    'libcudnn.so.9': 'libcudnn9-cuda-12=9.3.0.75-1', 'libmpi.so.40': 'libopenmpi3',
    'libcudart.so.12': 'cuda-cudart-12-6', 'libcublas.so.12': 'libcublas-12-6',
    'libcublasLt.so.12': 'libcublas-12-6', 'libcurand.so.10': 'libcurand-12-6',
    'libcufft.so.11': 'libcufft-12-6', 'libcusparse.so.12': 'libcusparse-12-6',
    'libcusolver.so.11': 'libcusolver-12-6', 'libnvrtc.so.12': 'cuda-nvrtc-12-6',
}
GPU_TEST = """
import json, torch
assert torch.version.cuda == '12.6', 'Beklenen NVIDIA CUDA 12.6 build bulunamadi'
assert torch.cuda.is_available(), 'PyTorch GPU yu kullanamiyor'
x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4,4)
product = x @ x
torch.cuda.synchronize()
assert torch.equal(product.cpu(), x.cpu() @ x.cpu()), 'GPU matris sonucu uyusmadi'
layer = torch.nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True, norm_first=True).cuda().eval()
with torch.inference_mode():
    y = layer(torch.ones(2,5,64,device='cuda'))
torch.cuda.synchronize()
assert bool(torch.isfinite(y).all().item()), 'Transformer GPU sonucu sonlu degil'
print(json.dumps({'cuda':True, 'torch':torch.__version__, 'cuda_build':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(0), 'cudnn':torch.backends.cudnn.version(),
                  'matmul_check':True, 'transformer_check':True}))
"""

def file_sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def validate_device(info):
    identity = info['identity']
    if identity.get('machine') != 'aarch64' or 'Orin' not in (identity.get('model') or ''):
        raise RuntimeError('Bu kurulum yalniz Jetson Orin ARM64 icindir')
    l4t = identity.get('l4t') or ''
    if not re.search(r'# R36\b', l4t) or not re.search(r'REVISION:\s*4\.', l4t):
        raise RuntimeError('Bu paket L4T 36.4.x icin hazirlandi; baska surum otomatik kurulmaz')
    if sys.version_info[:2] != (3,10):
        raise RuntimeError('NVIDIA wheel Python 3.10 gerektiriyor')
    version = info['current']['cuda_toolkit'].get('stdout', '')
    if not re.search(r'release 12\.6\b', version):
        raise RuntimeError('Kurulu CUDA 12.6 dogrulanamadi; surucu/JetPack degistirilmedi')


def download(url, target, expected_bytes, expected_sha=None):
    def valid(path):
        return (path.is_file() and path.stat().st_size == expected_bytes and
                (expected_sha is None or file_sha256(path) == expected_sha))
    if valid(target): return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    for attempt in range(2):
        try:
            print('Indiriliyor:', target.name, flush=True)
            count, next_notice = 0, 64*1024**2
            request = urllib.request.Request(url, headers={'User-Agent': 'specific-range-jetson'})
            with urllib.request.urlopen(request, timeout=45) as response, part.open('wb') as handle:
                while True:
                    block = response.read(1024**2)
                    if not block: break
                    count += len(block)
                    if count > expected_bytes: raise RuntimeError('Indirme beklenen boyutu asti')
                    handle.write(block)
                    if count >= next_notice:
                        print('Indirilen: {:.0f} / {:.0f} MB'.format(count/10**6, expected_bytes/10**6), flush=True)
                        next_notice += 64*1024**2
            if not valid(part): raise RuntimeError('Indirme boyut/SHA kontrolu basarisiz')
            part.replace(target)
            return target
        except Exception:
            if attempt == 1: raise
            print('Indirme bir kez yeniden deneniyor.', flush=True)


def unpack_sparse(archive, library):
    if file_sha256(archive) != SPARSE_SHA256:
        raise RuntimeError('NVIDIA cuSPARSELt arsiv SHA-256 uyusmadi')
    library.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive), 'r:xz') as tar:
        for member in tar.getmembers():
            if not member.name.startswith(SPARSE_PREFIX): continue
            name = member.name[len(SPARSE_PREFIX):]
            if '/' in name or not name.startswith('libcusparseLt.so'): continue
            if not (member.isfile() or member.issym() or member.islnk()): continue
            # Materialize library aliases as regular files in the private runtime.
            with tar.extractfile(member) as source:
                (library/name).write_bytes(source.read())
    if not (library/'libcusparseLt.so.0').is_file():
        raise RuntimeError('cuSPARSELt .so.0 dosyasi arsivde bulunamadi')


def write_wrapper(path, python, library):
    libraries = str(library) + ':/usr/local/cuda/lib64'
    text = ('#!/usr/bin/env bash\nexport LD_LIBRARY_PATH=' + shlex.quote(libraries) +
            '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec ' + shlex.quote(str(python)) + ' "$@"\n')
    path.write_text(text, encoding='utf-8')
    path.chmod(0o700)


def verify_gpu(wrapper):
    result = auto.query([str(wrapper), '-c', GPU_TEST], timeout=120)
    auto.save(auto.REPORT/'gpu_test_attempt.json', result)
    if result['code']:
        return None, result.get('stderr', '') + result.get('error', '')
    try:
        probe = json.loads(result['stdout'].strip().splitlines()[-1])
        if probe.get('cuda') and probe.get('matmul_check') and probe.get('transformer_check'):
            return probe, None
    except (KeyError, ValueError, IndexError): pass
    return None, 'GPU test sonucu okunamadi: ' + str(result)


def setup(base, repo):
    auto.phase('Cihaz ve CUDA surum kontrolu')
    info = auto.profile(base)
    validate_device(info)
    if shutil.disk_usage(str(base)).free < 6*1024**3:
        raise RuntimeError('GPU kurulumu icin en az 6 GiB bos disk gerekiyor')
    runtime = base/'pytorch_gpu_310_cuda126'
    python = runtime/'bin/python'
    if not python.is_file():
        if not auto.run([sys.executable, '-m', 'venv', str(runtime)], optional=True):
            auto.apt(['python3.10-venv'])
            auto.run([sys.executable, '-m', 'venv', str(runtime)])
    downloads = base/'nvidia-downloads'
    auto.phase('NVIDIA CUDA 12.6 PyTorch ve cuSPARSELt indirme')
    wheel = download(TORCH_URL, downloads/TORCH_FILE, TORCH_BYTES)
    with zipfile.ZipFile(str(wheel)) as z:
        if "'12.6'" not in z.read('torch/version.py').decode():
            raise RuntimeError('Wheel icindeki CUDA surumu 12.6 degil')
    sparse = download(SPARSE_URL, downloads/Path(SPARSE_URL).name, SPARSE_BYTES, SPARSE_SHA256)
    library = runtime/'nvidia-libs'
    unpack_sparse(sparse, library)
    auto.save(auto.REPORT/'vendor_downloads.json', {
        'torch_url': TORCH_URL, 'torch_bytes': wheel.stat().st_size,
        'torch_sha256': file_sha256(wheel), 'cusparselt_url': SPARSE_URL,
        'cusparselt_sha256': SPARSE_SHA256,
        'documentation': ['https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html',
                          'https://docs.pytorch.org/TensorRT/v2.6.1/getting_started/jetpack.html']})
    auto.phase('Ayri GPU Python ortamina kurulum')
    pip = [str(python), '-m', 'pip', '--disable-pip-version-check', '--no-input', 'install',
           '--only-binary=:all:', '--retries', '2', '--timeout', '30',
           '--cache-dir', str(base/'pip-cache'), '--index-url', 'https://pypi.org/simple']
    auto.run(pip + ['numpy==1.26.4'])
    auto.run(pip + [str(wheel)], timeout=3600)
    auto.run([str(python), '-m', 'pip', 'check'])
    wrapper = runtime/'gpu_python'
    write_wrapper(wrapper, python, library)
    auto.phase('Gercek GPU matris ve Transformer kontrolu')
    attempted = set()
    while True:
        probe, error = verify_gpu(wrapper)
        if probe: break
        package = next((package for lib, package in LIB_PACKAGES.items()
                        if lib in error and package not in attempted), None)
        if not package: raise RuntimeError(error)
        attempted.add(package)
        print('Eksik calisma kutuphanesi kurulacak:', package, flush=True)
        auto.apt([package])
    auto.save(auto.REPORT/'gpu_probe.json', probe)
    auto.save(auto.REPORT/'gpu_packages.json', auto.query([str(python), '-m', 'pip', 'freeze']))
    config = {'schema_version': 1, 'python': str(wrapper), 'version': [3,10],
              'runtime': str(runtime), 'identity': info['identity'], 'verified_utc': auto.stamp(),
              'probe': probe, 'torch_url': TORCH_URL}
    auto.save(base/'gpu_runtime.json', config)
    auto.save(auto.REPORT/'gpu_runtime.json', config)
    auto.STATE.update({'gpu': 'ready', 'stage': 'GPU PyTorch hazir'})
    print('GPU PYTORCH HAZIR:', probe['gpu'], '| torch:', probe['torch'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-publish', action='store_true')
    args = parser.parse_args()
    if os.name != 'posix': raise RuntimeError('Bu kurulum Jetson Linux icindir')
    repo = args.repo.resolve()
    base = Path.home()/'specific-range-jetson-auto'
    auto.PROJECT = repo
    auto.REPORT = base/('run_gpu_setup_' + auto.stamp())/'output'
    auto.REPORT.mkdir(parents=True, exist_ok=False)
    auto.STATE.update({'kind': 'gpu_setup', 'cpu': 'not_requested', 'gpu': 'not_run'})
    log = (auto.REPORT/'kurulum.log').open('a', encoding='utf-8', buffering=1)
    class Tee:
        def write(self, value):
            auto.TERMINAL.write(value)
            log.write(value)
            return len(value)
        def flush(self):
            auto.TERMINAL.flush()
            log.flush()
    sys.stdout = sys.stderr = Tee()
    try:
        setup(base, repo)
    except BaseException as exc:
        auto.STATE.update({'gpu': 'setup_failed', 'error': str(exc) or type(exc).__name__})
        traceback.print_exc()
    finally:
        try: auto.deliver(base, log)
        except BaseException:
            sys.stdout = sys.stderr = auto.TERMINAL
            traceback.print_exc()
            print('Yerel GPU kurulum kayitlari:', auto.REPORT)
    if not args.no_publish and auto.STATE.get('error') != 'KeyboardInterrupt':
        try:
            subprocess.call([sys.executable, str(repo/'scripts/publish_jetson_results.py'),
                             '--repo', str(repo), '--output', str(auto.REPORT)], timeout=240)
        except Exception as exc: print('Kurulum kaydi yerelde korundu:', exc)
    return 0 if auto.STATE['gpu'] == 'ready' else 1


if __name__ == '__main__': sys.exit(main())
