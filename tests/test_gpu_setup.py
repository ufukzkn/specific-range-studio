"""GPU runtime registration and a CUDA-only setup which cannot silently pass CPU."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_auto_test', ROOT/'scripts/jetson_auto.py')
AUTO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUTO)
spec = importlib.util.spec_from_file_location('gpu_setup_test', ROOT/'scripts/setup_jetson_torch.py')
GPU = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'jetson_auto': AUTO}):
    spec.loader.exec_module(GPU)


class GpuSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        AUTO.REPORT = self.base/'output'
        AUTO.REPORT.mkdir()
        self.identity = {'machine': 'aarch64', 'model': 'NVIDIA Jetson Orin Nano Super',
                         'l4t': '# R36 (release), REVISION: 4.7, BOARD: generic'}
        self.wrapper = self.base/'gpu/gpu_python'
        self.wrapper.parent.mkdir()
        self.wrapper.write_text('#!/bin/bash\n')

    def tearDown(self):
        self.temp.cleanup()

    def register(self):
        record = {'schema_version': 1, 'python': str(self.wrapper), 'version': [3,10],
                  'identity': self.identity, 'probe': {'cuda': True}}
        (self.base/'gpu_runtime.json').write_text(json.dumps(record))
        return record

    def test_registered_cuda_runtime_is_selected_with_its_exact_wrapper(self):
        self.register()
        result = AUTO.registered_gpu_runtime(self.base, self.identity)
        self.assertTrue(result['registered_runtime'])
        self.assertEqual(result['path'], str(self.wrapper))

    def test_full_runner_uses_registered_gpu_venv_instead_of_creating_nested_venv(self):
        self.register()
        AUTO.REPO, AUTO.SOURCE = ROOT, None
        info = {'identity': self.identity, 'cache_reused': False,
                'interpreters': [{'path': '/usr/bin/python3', 'version': [3,10,12]}]}
        cpu = self.base/'cpu/bin/python'
        def torch_probe(executable):
            return {'cuda': str(executable) == str(self.wrapper)}
        with patch.object(AUTO, 'profile', return_value=info), \
             patch.object(AUTO, 'find_output_usb', return_value=None), \
             patch.object(AUTO.os, 'chdir'), \
             patch.object(AUTO, 'query', return_value={'code': 0, 'stdout': '', 'stderr': ''}), \
             patch.object(AUTO, 'make_env', return_value=cpu) as make_env, \
             patch.object(AUTO, 'run', return_value=True), \
             patch.object(AUTO, 'torch_probe', side_effect=torch_probe), \
             patch.object(AUTO, 'benchmark', return_value=True) as benchmark:
            AUTO.main(self.base)
        self.assertEqual(make_env.call_count, 1)
        self.assertEqual(make_env.call_args[0][-1], 'cpu')
        gpu_calls = [call.args for call in benchmark.call_args_list if call.args[1] == 'gpu']
        self.assertEqual(len(gpu_calls), 2)
        self.assertTrue(all(call[0] == self.wrapper for call in gpu_calls))
        self.assertEqual(AUTO.STATE['gpu'], 'passed')

    def test_runtime_for_another_jetson_release_is_not_reused(self):
        self.register()
        changed = dict(self.identity, l4t='# R36 (release), REVISION: 5.0')
        self.assertIsNone(AUTO.registered_gpu_runtime(self.base, changed))

    def test_runtime_outside_private_workspace_is_rejected(self):
        record = self.register()
        record['python'] = sys.executable
        (self.base/'gpu_runtime.json').write_text(json.dumps(record))
        self.assertIsNone(AUTO.registered_gpu_runtime(self.base, self.identity))

    def test_wrong_cuda_is_rejected_before_any_install(self):
        info = {'identity': self.identity, 'current': {'cuda_toolkit': {'stdout': 'release 12.2'}}}
        with patch.object(GPU.sys, 'version_info', (3,10,12)):
            with self.assertRaisesRegex(RuntimeError, 'CUDA 12.6'):
                GPU.validate_device(info)
            info['current']['cuda_toolkit']['stdout'] = 'Cuda compilation tools, release 12.6, V12.6.68'
            GPU.validate_device(info)

    def test_failed_gpu_math_does_not_register_a_success(self):
        probe = {'cuda': True, 'matmul_check': False, 'transformer_check': True}
        with patch.object(AUTO, 'query', return_value={'code': 0, 'stdout': json.dumps(probe), 'stderr': ''}):
            ready, error = GPU.verify_gpu(self.wrapper)
        self.assertIsNone(ready)
        self.assertTrue(error)
        self.assertFalse((self.base/'gpu_runtime.json').exists())

    def test_ready_requires_both_real_cuda_tests(self):
        probe = {'cuda': True, 'matmul_check': True, 'transformer_check': True}
        with patch.object(AUTO, 'query', return_value={'code': 0, 'stdout': json.dumps(probe), 'stderr': ''}):
            ready, error = GPU.verify_gpu(self.wrapper)
        self.assertEqual(ready, probe)
        self.assertIsNone(error)

    def test_changed_vendor_library_archive_is_rejected(self):
        archive = self.base/'bad.tar.xz'
        archive.write_bytes(b'changed vendor data')
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            GPU.unpack_sparse(archive, self.base/'lib')
        self.assertFalse((self.base/'lib').exists())


if __name__ == '__main__': unittest.main()
