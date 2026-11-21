import base64,struct,zlib
import numpy as np

def png(pixels):
    a = np.ascontiguousarray(np.rint(np.clip(pixels, 0, 255)), dtype='u1')
    h, w = a.shape[:2]
    def chunk(tag, body):
        return struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body) & 0xffffffff)
    scan = b''.join(b'\0' + line.tobytes() for line in a)
    payload = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0 if a.ndim == 2 else 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(scan)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(payload).decode('ascii')

def points(mesh, r, c):
    r, c = np.broadcast_arrays(r, c)
    i = np.minimum(np.floor(r).astype(int), len(mesh)-2)
    j = np.minimum(np.floor(c).astype(int), mesh.shape[1]-2)
    u, v = (r-i)[..., None], (c-j)[..., None]
    return (1-u)*((1-v)*mesh[i,j] + v*mesh[i,j+1]) + u*((1-v)*mesh[i+1,j] + v*mesh[i+1,j+1])
