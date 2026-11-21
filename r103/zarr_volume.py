"""Reader for uint8 OME-Zarr v2 in the original array coordinate system."""
from itertools import product
import gzip,json,zlib
from pathlib import Path
import numpy as np


def crop_array(volume,origin):
    """Read a local uint8 OME-Zarr v2 level-zero crop in Z,Y,X order."""
    volume=Path(volume)
    attributes=json.loads((volume/'.zattrs').read_text())
    scale=attributes['multiscales'][0]
    axes=[x['name'] if isinstance(x,dict) else x for x in scale['axes']]
    if axes!=['z','y','x']:raise ValueError('Expected Z,Y,X axes at native resolution.')
    path=Path(scale['datasets'][0]['path'])
    if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid level-zero dataset path.')
    level=volume/path
    meta=json.loads((level/'.zarray').read_text())
    if meta['zarr_format']!=2 or np.dtype(meta['dtype'])!=np.uint8 or len(meta['shape'])!=3:
        raise ValueError('Expected a local, 3D, uint8 Zarr v2 CT at level zero.')
    shape=np.array(meta['shape'],dtype=int)
    if origin is None:
        if tuple(shape)!=(128,128,128):raise ValueError('Supply a local 128-cubed volume in the same coordinate frame as the input meshes.')
        origin=np.zeros(3,dtype=int)
    origin=np.asarray(origin,dtype=int);end=origin+128
    if np.any(origin<0) or np.any(end>shape):raise ValueError('The 128-cubed crop falls outside the selected level.')
    cs=np.array(meta['chunks'],dtype=int)
    codec=meta.get('compressor') or {}
    if meta.get('filters') or codec.get('id') not in (None,'gzip','zlib'):
        try:import zarr
        except ImportError:raise ValueError('This store uses extra codecs. Install compatible zarr and numcodecs in the operator environment, or supply an uncompressed level-zero crop.')
        data=zarr.open_array(str(level),mode='r')[tuple(slice(int(a),int(b)) for a,b in zip(origin,end))]
        return np.ascontiguousarray(data),origin
    if np.any(cs<=0) or meta.get('order','C') not in ('C','F'):raise ValueError('Invalid chunk shape or storage order.')
    separator=meta.get('dimension_separator','.')
    if separator not in ('.','/'):raise ValueError('Invalid v2 dimension separator.')
    output=np.empty((128,)*3,dtype='u1')
    expected=int(np.prod(cs,dtype=object))
    for idx in product(*(range(int(a//k),int((b-1)//k+1)) for a,b,k in zip(origin,end,cs))):
        source=level/separator.join(map(str,idx))
        start=np.asarray(idx)*cs; lo=np.maximum(start,origin);hi=np.minimum(start+cs,end)
        dest=tuple(slice(int(a),int(b)) for a,b in zip(lo-origin,hi-origin))
        if not source.is_file():
            if meta.get('fill_value') is None:raise ValueError('Missing a chunk with an undefined fill value.')
            output[dest]=meta['fill_value'];continue
        if codec.get('id')=='gzip':
            with gzip.open(source,'rb') as f:data=f.read(expected+1)
        elif codec.get('id')=='zlib':
            d=zlib.decompressobj();data=d.decompress(source.read_bytes(),expected+1)
            if not d.eof:raise ValueError('Incomplete or oversized zlib chunk.')
        else:data=source.read_bytes()
        if len(data)!=expected:raise ValueError('Incorrect decoded Zarr chunk size.')
        chunk=np.frombuffer(data,dtype='u1').reshape(tuple(cs),order=meta['order'])
        output[dest]=chunk[tuple(slice(int(a),int(b)) for a,b in zip(lo-start,hi-start))]
    return output,origin
