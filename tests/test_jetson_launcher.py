"""Clone-only handoff, integrity, and unsupported-device failure behavior."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('jetson_auto', ROOT/'scripts/jetson_auto.py')
AUTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTO)


class JetsonLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        AUTO.REPORT = self.base/'output'
        AUTO.REPORT.mkdir()
        AUTO.REPO = ROOT
        AUTO.SOURCE = None

    def tearDown(self):
        self.temp.cleanup()

    def test_temporarily_unreadable_sysfs_sensor_does_not_enter_text_decoder(self):
        with patch.object(AUTO.Path, 'read_bytes', return_value=None), \
             patch.object(AUTO.Path, 'read_text', side_effect=TypeError("can't concat NoneType to bytes")):
            self.assertIsNone(AUTO.textfile('/sys/class/thermal/thermal_zone0/temp'))
        with patch.object(AUTO.Path, 'read_bytes', return_value=b'52000\n'):
            self.assertEqual(AUTO.textfile('/sys/class/thermal/thermal_zone0/temp'), '52000')

    def test_device_profile_survives_one_missing_temperature_reading(self):
        sensor = self.base/'thermal_zone0'
        sensor.mkdir()
        (sensor/'type').write_bytes(b'cpu-thermal\n')
        (sensor/'temp').write_bytes(b'52000\n')
        original = AUTO.Path.read_bytes
        def sensor_read(path):
            return None if path == sensor/'temp' else original(path)
        with patch.object(AUTO.Path, 'read_bytes', autospec=True, side_effect=sensor_read), \
             patch.object(AUTO.Path, 'glob', return_value=[sensor]), \
             patch.object(AUTO.shutil, 'which', return_value=sys.executable), \
             patch.object(AUTO, 'query', return_value={'code': 0, 'stdout': '[3,10,12]', 'stderr': ''}):
            info = AUTO.profile(self.base)
        self.assertIn('cpu-thermal', info['current']['thermal'])
        self.assertIsNone(info['current']['thermal']['cpu-thermal'])
        self.assertTrue((AUTO.REPORT/'device_profile.json').is_file())
        saved = json.loads((AUTO.REPORT/'device_profile.json').read_text())
        self.assertIsNone(saved['current']['thermal']['cpu-thermal'])

    def test_clone_snapshot_contains_frozen_models_and_source_provenance(self):
        target = self.base/'project'
        manifest = AUTO.prepare_repo(ROOT, target)
        self.assertEqual(manifest['experiment_label'], 'functional_check')
        self.assertTrue((target/'scripts/run_edge_benchmark.py').is_file())
        self.assertTrue((target/'artifacts/research_bundle_portable/ft_transformer/model.pt').is_file())
        record = json.loads((AUTO.REPORT/'runtime_source_manifest.json').read_text())
        self.assertEqual(record['source']['model_zip_sha256'], AUTO.BUNDLE_ZIP_SHA256)
        for name, expected in record['files'].items():
            self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(), expected)
        self.assertFalse((target/'.git').exists())
        self.assertFalse((target/'.venv').exists())
        self.assertNotIn('artifacts/benchmarks', str(record['files']))

    def test_changed_model_zip_is_rejected_before_extraction(self):
        corrupt = self.base/'changed.zip'
        corrupt.write_bytes(b'changed model data')
        target = self.base/'project'
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            AUTO.extract_repo_bundle(corrupt, target)
        self.assertFalse(target.exists())

    def test_missing_or_ambiguous_usb_is_optional(self):
        with patch.object(AUTO, 'walk', return_value=[]):
            self.assertIsNone(AUTO.find_output_usb())
        candidates = [(self.base/'one/jetsonicinolan', []),
                      (self.base/'two/jetsonicinolan', [])]
        with patch.object(AUTO, 'walk', return_value=candidates), patch.object(AUTO, 'mount_id') as mount:
            self.assertIsNone(AUTO.find_output_usb())
            mount.assert_not_called()

    def test_old_python_stops_with_profile_and_sources_before_install(self):
        info = {'identity': {'machine': 'aarch64', 'model': 'Jetson', 'l4t': 'R32'},
                'cache_reused': False, 'interpreters': [{'path': 'python3', 'version': [3, 6, 9]}]}
        with patch.object(AUTO, 'profile', return_value=info), \
             patch.object(AUTO, 'find_output_usb', return_value=None), \
             patch.object(AUTO.os, 'chdir'), patch.object(AUTO, 'apt') as apt, \
             patch.object(AUTO, 'make_env') as env:
            with self.assertRaisesRegex(RuntimeError, 'Python 3.10'):
                AUTO.main(self.base)
            apt.assert_not_called()
            env.assert_not_called()
        self.assertTrue((AUTO.REPORT/'repository_source.json').is_file())


if __name__ == '__main__':
    unittest.main()
