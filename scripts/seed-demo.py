"""Fill a dev backend (ideally started with --fake-engine) with demo data.

    python scripts/seed-demo.py --backend http://127.0.0.1:8765 --token dev

Creates a few characters from synthetic "singing" (harmonic tones with vibrato and
breaths), organizes them, starts a training run, and sets up a project with a song.
Only uses numpy, soundfile and the standard library.
"""

import argparse
import json
import math
import os
import tempfile
import time
import urllib.request

import numpy as np
import soundfile as sf

SR = 44100


def sing(seconds: float, base_midi: float, seed: int) -> np.ndarray:
    """A phrase of notes with vibrato, harmonics, breathy noise and pauses."""
    rng = np.random.default_rng(seed)
    out = []
    t_total = 0.0
    while t_total < seconds:
        note_len = rng.uniform(0.35, 1.1)
        midi = base_midi + rng.choice([0, 2, 4, 5, 7, 9, 12, -3, -5])
        n = int(note_len * SR)
        t = np.arange(n) / SR
        f0 = 440 * 2 ** ((midi - 69) / 12) * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
        phase = 2 * np.pi * np.cumsum(f0) / SR
        tone = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 9))
        env = np.minimum(1, t / 0.05) * np.minimum(1, (note_len - t) / 0.08)
        out.append(0.25 * tone * env + 0.004 * rng.standard_normal(n))
        t_total += note_len
        if rng.random() < 0.25:
            gap = int(rng.uniform(0.25, 0.6) * SR)
            out.append(0.002 * rng.standard_normal(gap))
            t_total += gap / SR
    return np.concatenate(out)[: int(seconds * SR)].astype(np.float32)


class Api:
    def __init__(self, base: str, token: str):
        self.base = base.rstrip('/') + '/api'
        self.token = token

    def call(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header('Authorization', f'Bearer {self.token}')
        if data is not None:
            req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read()
            return json.loads(raw) if raw else None

    def wait(self, job, timeout=600):
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.call('GET', f"/jobs/{job['id']}")
            if j['status'] in ('succeeded', 'failed', 'cancelled'):
                if j['status'] != 'succeeded':
                    print('  job', j['type'], j['status'], j.get('error'))
                return j
            time.sleep(0.5)
        raise TimeoutError(job['id'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='http://127.0.0.1:8765')
    ap.add_argument('--token', default='dev')
    ap.add_argument('--train', action='store_true', help='also start a (fake) training run')
    args = ap.parse_args()
    api = Api(args.backend, args.token)
    tmp = tempfile.mkdtemp(prefix='stellaria-demo-')

    voices = [('小雪', 64, '#9aaabd', ['原创']), ('阿岚', 57, '#c9a6a0', ['朋友']), ('星野', 60, '#cdbb9f', ['原创', '男声'])]
    created = []
    for i, (name, base, color, tags) in enumerate(voices):
        c = api.call('POST', '/characters', {'name': name, 'color': color, 'tags': tags})
        api.call('PATCH', f"/characters/{c['id']}", {'consent': {'source': '本人录制', 'authorized': True, 'note': ''}})
        paths = []
        for k in range(3):
            p = os.path.join(tmp, f'{name}_录音{k + 1}.wav')
            sf.write(p, sing(240 + 60 * k, base, seed=i * 10 + k), SR)
            paths.append(p)
        job = api.call('POST', f"/characters/{c['id']}/raw", {'paths': paths})
        api.wait(job)
        if i < 2:
            job = api.call('POST', f"/characters/{c['id']}/organize", {})
            api.wait(job)
        created.append(c)
        print('character', name, c['id'])

    if args.train:
        c = created[0]
        m = api.call('POST', '/models', {'name': c['name'], 'characterIds': [c['id']], 'config': {'targetSteps': 20000, 'intervalVal': 2000}})
        api.call('POST', f"/models/{m['id']}/train", {})
        print('training', m['id'])

    p = api.call('POST', '/projects', {'name': '夏日小调'})
    song = os.path.join(tmp, '夏日小调.wav')
    sf.write(song, sing(48, 62, seed=99), SR)
    job = api.call('POST', f"/projects/{p['id']}/media", {'paths': [song]})
    api.wait(job)
    print('project', p['id'])


if __name__ == '__main__':
    main()
