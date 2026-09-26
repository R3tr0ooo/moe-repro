import ast
import json
import os
from pathlib import Path
import pickletools
import re
import zipfile
ROOT = Path(__file__).resolve().parent
PATTERNS = {'absolute_windows_path': re.compile('\\b[A-Za-z]:[\\\\/]+'), 'home_or_mount_path': re.compile('/(?:Users|home|mnt/[a-z])/[^\\s]+'), 'device_uuid': re.compile('GPU-[a-fA-F0-9-]{36}'), 'private_ip': re.compile('\\b(?:192\\.168\\.\\d{1,3}\\.\\d{1,3}|10\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}|172\\.(?:1[6-9]|2\\d|3[01])\\.\\d{1,3}\\.\\d{1,3})\\b'), 'access_token': re.compile('\\b(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{24,})\\b')}

def main():
    findings = []
    files, total_bytes = (0, 0)
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or '.git' in path.relative_to(ROOT).parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        files += 1
        total_bytes += path.stat().st_size
        if any((part in {'__pycache__', '.env'} for part in path.relative_to(ROOT).parts)):
            findings.append({'file': relative, 'rule': 'private_or_generated_directory'})
        if path.suffix in {'.py', '.json', '.md', '.txt', '.csv'}:
            content = path.read_text(encoding='utf-8-sig')
            if path.suffix == '.py':
                ast.parse(content, filename=relative)
            elif path.suffix == '.json':
                json.loads(content)
        elif path.suffix == '.pt':
            strings = []
            with zipfile.ZipFile(path) as archive:
                strings.extend(archive.namelist())
                for name in archive.namelist():
                    if name.endswith('.pkl'):
                        for opcode, value, _ in pickletools.genops(archive.read(name)):
                            if isinstance(value, str):
                                strings.append(value)
            content = '\n'.join(strings)
        else:
            findings.append({'file': relative, 'rule': 'unreviewed_file_type'})
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append({'file': relative, 'rule': label})
        for key in ('USERNAME', 'COMPUTERNAME'):
            value = os.environ.get(key, '')
            if len(value) >= 4 and value.casefold() in content.casefold():
                findings.append({'file': relative, 'rule': 'host_identifier'})
    print(json.dumps({'status': 'pass' if not findings else 'fail', 'files': files, 'bytes': total_bytes, 'findings': findings}))
    if findings:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
