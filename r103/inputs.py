import gzip, math, pathlib, re
import numpy as np
MAX_FILE=40*1024**2
def require(ok,why):
    if not ok:raise ValueError(why)

def nrrd(path):
    p=pathlib.Path(path);require(p.stat().st_size<=MAX_FILE,'NRRD_FILE_SIZE')
    raw=p.read_bytes();sep=re.search(rb'\r?\n\r?\n',raw[:65536]);require(sep is not None,'NRRD_HEADER')
    lines=raw[:sep.start()].decode('ascii').splitlines();require(lines[0].startswith('NRRD000'),'NRRD_MAGIC')
    fields={}
    for line in lines[1:]:
        if not line.startswith('#') and ':' in line:
            k,v=line.split(':',1);fields[k.lower().strip()]=v.strip()
    require('data file' not in fields and 'datafile' not in fields,'NRRD_DETACHED_UNSUPPORTED')
    require(fields.get('dimension')=='3','NRRD_DIMENSION')
    sizes=tuple(map(int,fields['sizes'].split()));require(len(sizes)==3 and max(sizes)<=512 and min(sizes)>1,'NRRD_SHAPE')
    types={'uchar':'u1','uint8':'u1','unsigned char':'u1','ushort':'u2','uint16':'u2','unsigned short':'u2',
      'short':'i2','int16':'i2','uint':'u4','unsigned int':'u4','uint32':'u4','int':'i4','int32':'i4','float':'f4','double':'f8'}
    require(fields.get('type') in types,'NRRD_TYPE')
    dtype=np.dtype(types[fields['type']]);require(fields.get('endian','little') in ('little','big'),'NRRD_ENDIAN')
    if dtype.itemsize>1:dtype=dtype.newbyteorder('<' if fields.get('endian','little')=='little' else '>')
    expected=math.prod(sizes)*dtype.itemsize;require(expected<=128*1024**2,'NRRD_ARRAY_LIMIT')
    payload=raw[sep.end():];encoding=fields.get('encoding','raw').lower()
    if encoding in ('gzip','gz'):
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as f:payload=f.read(expected+1)
    else:require(encoding=='raw','NRRD_ENCODING')
    require(len(payload)==expected,'NRRD_PAYLOAD_SIZE')
    # NRRD's first axis varies fastest: XYZ sizes -> NumPy ZYX.
    a=np.frombuffer(payload,dtype=dtype).reshape(sizes[::-1]).astype(dtype.newbyteorder('='),copy=True)
    require(np.isfinite(a).all(),'NRRD_NONFINITE')
    return a,fields

def neutral_seed(ct):
    """The center is the only annotation-provided location; orientation uses only local CT."""
    if ct.shape!=(128,128,128) or ct.dtype!=np.uint8:raise ValueError('Expected a 128-cubed uint8 CT crop.')
    patch=ct[56:73,56:73,56:73].astype(np.float64)
    for ax in range(3):
        pads=[(0,0)]*3;pads[ax]=(1,1)
        p=np.moveaxis(np.pad(patch,pads,mode='edge'),ax,0)
        patch=np.moveaxis((p[:-2]+2*p[1:-1]+p[2:])/4.,0,ax)
    grad=np.stack(np.gradient(patch),axis=-1)[2:-2,2:-2,2:-2].reshape(-1,3)
    grid=np.indices((13,13,13),dtype=float)-6
    weights=np.exp(-np.sum(grid**2,axis=0).ravel()/18.)
    tensor=grad.T@(weights[:,None]*grad)
    eig,axes=np.linalg.eigh(tensor)
    # Deterministic fallback on an isotropic local tensor, still giving every method the same edge.
    normal=np.array([1.,0.,0.]) if eig[-1]-eig[-2]<=1e-10*max(float(eig[-1]),1.) else axes[:,-1]
    if normal[np.argmax(np.abs(normal))]<0:normal=-normal
    tangent=np.eye(3)[int(np.argmin(np.abs(normal)))].copy()
    tangent-=normal*np.dot(tangent,normal);tangent/=np.linalg.norm(tangent)
    width=np.cross(normal,tangent)
    return (np.array([64.,64.,64.])+np.arange(-2,1)[:,None,None]*2.*tangent
            +np.arange(-4,5)[None,:,None]*1.5*width).astype('<f4')
