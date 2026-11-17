"""Partition-selection safety and real local Git result publication."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/(name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


USB = module('jetson_usb_check')
UPLOAD = module('publish_jetson_results')


class UsbSelectionTests(unittest.TestCase):
    def test_sudo_prompt_supports_nonseekable_terminal(self):
        class Tty(io.RawIOBase):
            def readable(self): return True
            def writable(self): return True
            def seekable(self): return False
        def terminal_open(path, mode, **kwargs):
            self.assertEqual(path, '/dev/tty')
            if mode == 'r+': return io.BufferedRandom(Tty())
            self.assertIn(mode, ['rb', 'wb'])
            self.assertEqual(kwargs.get('buffering'), 0)
            return Tty()
        with patch.object(USB.os, 'geteuid', return_value=1000, create=True), \
             patch.object(USB.shutil, 'which', return_value='/usr/bin/sudo'), \
             patch('builtins.open', side_effect=terminal_open), \
             patch.object(USB.subprocess, 'call', return_value=0) as call, \
             patch.object(USB, 'query', return_value={'code': 1}) as cached:
            self.assertTrue(USB.sudo_ready())
            cached.assert_not_called()
        self.assertEqual(call.call_args[0][0], ['sudo', '-v'])
        self.assertEqual(USB.SUDO_AUTH['method'], 'controlling_terminal')

    def test_small_metadata_partition_does_not_hide_real_exfat_partition(self):
        disk = {'name': '/dev/sda', 'type': 'disk', 'size': 64000000000, 'tran': 'usb', 'rm': True,
                'children': [{'name': '/dev/sda1', 'type': 'part', 'size': 32768, 'fstype': None},
                             {'name': '/dev/sda2', 'type': 'part', 'size': 63900000000, 'fstype': 'exfat'}]}
        candidates = USB.exfat_candidates(USB.usb_disks({'blockdevices': [disk]}))
        self.assertEqual([n['name'] for n in candidates], ['/dev/sda2'])

    def test_usb_boot_disk_and_internal_disk_are_excluded(self):
        system = {'name': '/dev/sda', 'type': 'disk', 'tran': 'usb', 'children': [
            {'name': '/dev/sda1', 'mountpoint': '/', 'fstype': 'ext4'}]}
        internal = {'name': '/dev/nvme0n1', 'type': 'disk', 'tran': 'nvme', 'rm': False}
        self.assertEqual(USB.usb_disks({'blockdevices': [system, internal]}), [])

    def test_tiny_disk_is_not_automatically_mounted(self):
        self.assertEqual(USB.exfat_candidates([{'name': '/dev/sda', 'fstype': 'exfat', 'size': 32768}]), [])

    def test_mount_request_is_read_only_and_uses_detected_partition(self):
        candidate = {'name': '/dev/sdz2', 'fstype': 'exfat', 'size': 8000000000}
        with patch.object(USB.shutil, 'which', return_value='/usr/bin/udisksctl'), \
             patch.object(USB, 'query', return_value={'code': 0}) as query:
            self.assertEqual(USB.try_mount(candidate, False)['status'], 'mounted_read_only')
        self.assertEqual(query.call_args[0][0], ['udisksctl', 'mount', '--block-device', '/dev/sdz2',
                                               '--options', 'ro', '--no-user-interaction'])

    def test_kernel_exfat_error_falls_back_to_installed_fuse_with_sudo(self):
        candidate = {'name': '/dev/sdz1', 'fstype': 'exfat', 'size': 61523984384, 'label': 'usb'}
        disk = {'name': '/dev/sdz', 'type': 'disk', 'tran': 'usb', 'children': [candidate]}
        def executable(name):
            return '/usr/bin/' + name if name in ['udisksctl', 'mount.exfat-fuse'] else None
        with patch.object(USB.shutil, 'which', side_effect=executable), \
             patch.object(USB.os, 'geteuid', return_value=1000, create=True), \
             patch.object(USB.os, 'getuid', return_value=1000, create=True), \
             patch.object(USB.os, 'getgid', return_value=1000, create=True), \
             patch.object(USB.getpass, 'getuser', return_value='test-user'), \
             patch.object(USB, 'read_layout', return_value=({'blockdevices': [disk]}, {})), \
             patch.object(USB, 'query', side_effect=[{'code': 1, 'stderr': "unknown filesystem type 'exfat'"},
                                                   {'code': 0}, {'code': 0}]) as query:
            result = USB.try_mount(candidate, True)
        self.assertEqual(result['status'], 'mounted_read_only')
        command = query.call_args_list[-1][0][0]
        self.assertEqual(command[:3], ['sudo', '-n', '/usr/bin/mount.exfat-fuse'])
        self.assertIn('ro,uid=1000,gid=1000', command)
        self.assertEqual(command[-2], '/dev/sdz1')


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.remote = self.base/'remote.git'
        subprocess.run(['git', 'init', '--bare', str(self.remote)], check=True, capture_output=True)
        self.seed = self.base/'seed'
        subprocess.run(['git', 'clone', str(self.remote), str(self.seed)], check=True, capture_output=True)
        self.g('config', 'user.name', 'Test runner')
        self.g('config', 'user.email', 'test@localhost')
        self.g('config', 'commit.gpgsign', 'false')
        self.g('checkout', '-b', 'jetson-benchmark')
        (self.seed/'source.txt').write_text('original source')
        self.g('add', 'source.txt')
        self.g('commit', '-m', 'Initial source')
        self.g('push', 'origin', 'jetson-benchmark')
        self.output = self.base/'run_test/output'
        self.output.mkdir(parents=True)
        (self.output/'STATUS.json').write_text('{"cpu":"passed"}')
        (self.seed/'source.txt').write_text('uncommitted user change')

    def tearDown(self):
        # Git object files can be read-only on Windows.
        for path in self.base.rglob('*'):
            if path.is_file(): path.chmod(0o600)
        self.temp.cleanup()

    def g(self, *args):
        return subprocess.run(['git'] + list(args), cwd=str(self.seed), capture_output=True,
                              text=True, check=True).stdout.strip()

    def test_results_are_committed_and_pushed_without_source_changes(self):
        receipt = UPLOAD.publish(self.seed, [self.output], self.base, remote=str(self.remote))
        self.assertEqual(receipt['status'], 'pushed')
        show = subprocess.run(['git', '--git-dir', str(self.remote), 'show',
                               'jetson-benchmark:jetson-results/run_test/STATUS.json'],
                              capture_output=True, text=True, check=True).stdout
        self.assertEqual(json.loads(show), {'cpu': 'passed'})
        original = subprocess.run(['git', '--git-dir', str(self.remote), 'show',
                                   'jetson-benchmark:source.txt'], capture_output=True, text=True, check=True).stdout
        self.assertEqual(original, 'original source')
        self.assertEqual((self.seed/'source.txt').read_text(), 'uncommitted user change')
        self.assertTrue(self.g('status', '--short'))

    def test_failed_push_preserves_local_commit_and_report(self):
        original = UPLOAD.git
        def deny_push(args, *rest, **kwargs):
            if args[0] == 'push': raise RuntimeError('write permission denied')
            return original(args, *rest, **kwargs)
        with patch.object(UPLOAD, 'git', side_effect=deny_push):
            receipt = UPLOAD.publish(self.seed, [self.output], self.base, remote=str(self.remote))
        self.assertEqual(receipt['status'], 'committed_locally')
        self.assertTrue((self.output/'STATUS.json').is_file())
        self.assertTrue((Path(receipt['workspace'])/'.git').is_dir())
        saved = json.loads((self.output.parent/'github_upload.json').read_text())
        self.assertEqual(saved['commit'], receipt['commit'])

    def test_network_failure_before_clone_leaves_upload_status_next_to_report(self):
        managed = self.base/'specific-range-jetson-auto/run_one/output'
        managed.mkdir(parents=True)
        (managed/'STATUS.json').write_text('{}')
        with patch.object(UPLOAD.Path, 'home', return_value=self.base), \
             patch.object(sys, 'argv', ['publish_jetson_results.py', '--repo', str(self.seed)]), \
             patch.object(UPLOAD, 'publish', side_effect=RuntimeError('offline')):
            self.assertEqual(UPLOAD.main(), 1)
        status = json.loads((managed.parent/'github_upload.json').read_text())
        self.assertEqual(status['status'], 'upload_failed')
        self.assertIn('offline', status['error'])


if __name__ == '__main__': unittest.main()
