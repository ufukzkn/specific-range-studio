sudo -v && python3 -u - <<'JETSON_SAAT_PY'
import http.client
import itertools
import subprocess
import sys
import time
import uuid
from email.utils import parsedate_to_datetime

HOSTS = ('www.google.com', 'www.cloudflare.com', 'ubuntu.com')

def internet_saati(host, secure=False):
    cls = http.client.HTTPSConnection if secure else http.client.HTTPConnection
    conn = cls(host, timeout=10)
    try:
        conn.request('HEAD', '/?jetson_clock=' + uuid.uuid4().hex,
                     headers={'Cache-Control': 'no-cache'})
        response = conn.getresponse()
        value = parsedate_to_datetime(response.getheader('Date', ''))
        if value.tzinfo is None or not 2024 <= value.year <= 2099:
            raise ValueError('Gecersiz tarih')
        if float(response.getheader('Age', '0')) > 30:
            raise ValueError('Eski onbellek yaniti')
        return value.timestamp(), time.monotonic()
    finally:
        conn.close()

def komut(*args):
    subprocess.check_call(['sudo', '-n'] + list(args))

try:
    samples = []
    for host in HOSTS:
        try:
            stamp, tick = internet_saati(host)
            samples.append((stamp, tick))
            print(host + ': saat alindi')
        except Exception as exc:
            print(host + ': ' + str(exc))
    now = time.monotonic()
    dates = [stamp + now - tick for stamp, tick in samples]
    pair = next((p for p in itertools.combinations(dates, 2)
                 if abs(p[0] - p[1]) <= 120), None)
    if pair is None:
        raise RuntimeError('Iki sunucudan uyumlu saat alinamadi. Internet/Wi-Fi girisini kontrol et.')
    target = max(pair)
    komut('timedatectl', 'set-ntp', 'false')
    komut('date', '-u', '-s', '@' + str(int(target)))
    verified = False
    for host in HOSTS:
        try:
            stamp, tick = internet_saati(host, secure=True)
            if abs(stamp - time.time()) > 120:
                raise ValueError('Saat dogrulamasi uyusmadi')
            verified = True
            print('SAAT HTTPS ILE DOGRULANDI: ' + host)
            break
        except Exception as exc:
            print('HTTPS ' + host + ': ' + str(exc))
    if not verified:
        raise RuntimeError('Saat HTTPS ile dogrulanamadi; kuruluma gecilmedi.')
    subprocess.call(['sudo', '-n', 'timedatectl', 'set-timezone', 'Europe/Istanbul'])
    subprocess.call(['sudo', '-n', 'timedatectl', 'set-ntp', 'true'])
    subprocess.call(['date'])
    komut('apt-get', 'update')
    komut('apt-get', 'install', '-y', 'exfat-fuse')
    print('\nTAMAM. USB yi cikarip yeniden tak ve jetsonicinolan klasorunu ac.')
except Exception as exc:
    print('\nDURDU: ' + str(exc))
    print('Bu son mesajlari paylas; bir sonraki adimi ona gore belirleyelim.')
    sys.exit(1)
JETSON_SAAT_PY
