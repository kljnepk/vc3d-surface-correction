"""Small, local coordinate-TIFF and OME-Zarr v2 adapters for the benchmark."""
import json
import struct
from pathlib import Path

import numpy as np


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def image8(raw):
    lo, hi = np.percentile(raw[::3,::3,::3], [2,98])
    if not hi > lo:
        raise ValueError('The CT normalization range is empty.')
    return np.rint(255*np.clip((raw.astype(np.float64)-lo)/(hi-lo), 0., 1.)).astype(np.uint8)


def write_coordinate(path, grid):
    grid=np.ascontiguousarray(grid, dtype='<f4')
    rows, cols=grid.shape
    tags={256:(4,cols), 257:(4,rows), 258:(3,32), 259:(3,1), 262:(3,1),
          273:(4,0), 274:(3,1), 277:(3,1), 278:(4,rows), 279:(4,grid.nbytes),
          284:(3,1), 339:(3,3)}
    data_offset=8+2+12*len(tags)+4
    tags[273]=(4,data_offset)
    with path.open('wb') as stream:
        stream.write(b'II'+struct.pack('<HIH',42,8,len(tags)))
        for tag,(kind,val) in sorted(tags.items()):
            stream.write(struct.pack('<HHII',tag,kind,1,val))
        stream.write(struct.pack('<I',0))
        stream.write(grid.tobytes())


def read_system_tiff(path):
    import ctypes as c
    from ctypes.util import find_library
    library=find_library('tiff')
    if not library:
        return None
    lib=c.CDLL(library)
    lib.TIFFOpen.argtypes=(c.c_char_p,c.c_char_p)
    lib.TIFFOpen.restype=c.c_void_p
    lib.TIFFClose.argtypes=(c.c_void_p,)
    lib.TIFFGetFieldDefaulted.argtypes=(c.c_void_p,c.c_uint32)
    lib.TIFFReadScanline.argtypes=(c.c_void_p,c.c_void_p,c.c_uint32,c.c_uint16)
    lib.TIFFIsTiled.argtypes=(c.c_void_p,)
    lib.TIFFScanlineSize.argtypes=(c.c_void_p,)
    lib.TIFFScanlineSize.restype=c.c_ssize_t
    lib.TIFFTileSize.argtypes=(c.c_void_p,)
    lib.TIFFTileSize.restype=c.c_ssize_t
    lib.TIFFReadTile.argtypes=(c.c_void_p,c.c_void_p,c.c_uint32,c.c_uint32,c.c_uint32,c.c_uint16)
    lib.TIFFReadTile.restype=c.c_ssize_t
    handle=lib.TIFFOpen(bytes(path),b'r')
    if not handle:
        return None
    try:
        def tag(key,kind):
            result=kind()
            if lib.TIFFGetFieldDefaulted(handle,key,c.byref(result))!=1:
                return 0
            return result.value
        w,h=tag(256,c.c_uint32),tag(257,c.c_uint32)
        bits,fmt=tag(258,c.c_uint16),tag(339,c.c_uint16)
        samples=tag(277,c.c_uint16)
        dt={(32,3):'f4',(8,1):'u1',(16,1):'u2'}.get((bits,fmt))
        if (not dt or not 0<w<=4096 or not 0<h<=4096 or samples not in (1,3,4)
                or tag(274,c.c_uint16)!=1 or tag(284,c.c_uint16)!=1):
            return None
        data=np.empty((h,w,samples),dtype=dt)
        if lib.TIFFIsTiled(handle):
            tw,th=tag(322,c.c_uint32),tag(323,c.c_uint32)
            if not 0<tw<=4096 or not 0<th<=4096:
                return None
            tile=np.empty((th,tw,samples),dtype=dt)
            if lib.TIFFTileSize(handle)!=tile.nbytes:
                return None
            for y in range(0,h,th):
                for x in range(0,w,tw):
                    if lib.TIFFReadTile(handle,tile.ctypes.data,x,y,0,0)<0:
                        raise ValueError('Could not read a TIFF tile.')
                    hh,ww=min(h-y,th),min(w-x,tw)
                    data[y:y+hh,x:x+ww]=tile[:hh,:ww]
        else:
            if lib.TIFFScanlineSize(handle)!=data[0].nbytes:
                return None
            for y in range(h):
                if lib.TIFFReadScanline(handle,data[y].ctypes.data,y,0)!=1:
                    raise ValueError('Could not read a TIFF scanline.')
        return data[:,:,0] if samples==1 else data
    finally:
        lib.TIFFClose(handle)


def read_tiff(path):
    # Prefer installed full decoders for downstream TIFF encodings such as LZW.
    for backend in ('tifffile','pillow','opencv'):
        try:
            if backend=='tifffile':
                import tifffile
                return tifffile.imread(path)
            if backend=='pillow':
                from PIL import Image
                with Image.open(path) as im:
                    return np.array(im)
            import cv2
            array=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if array is not None:
                return array
        except (ImportError, OSError, ValueError, RuntimeError, KeyError):
            continue
    try:
        decoded=read_system_tiff(path)
        if decoded is not None:
            return decoded
    except (OSError,AttributeError,ValueError):
        pass
    data=path.read_bytes()
    if data[:2] not in (b'II',b'MM'):
        raise ValueError('Invalid TIFF header.')
    endian='<' if data[:2]==b'II' else '>'
    def field(form,off):
        return struct.unpack_from(endian+form,data,off)[0]
    if field('H',2)!=42:
        raise ValueError('BigTIFF requires a compatible TIFF reader.')
    off=field('I',4)
    tags={}
    for i in range(field('H',off)):
        p=off+2+12*i
        key,typ,count=struct.unpack_from(endian+'HHI',data,p)
        if count==1 and typ in (3,4):
            tags[key]=field('H' if typ==3 else 'I',p+8)
    if (tags.get(259,1)!=1 or tags.get(277,1)!=1 or 273 not in tags
            or tags.get(278,0)<tags[257] or tags.get(274,1)!=1):
        raise ValueError('This TIFF requires a compatible decoder such as tifffile with imagecodecs.')
    dtype={(32,3):'f4',(8,1):'u1',(16,1):'u2'}.get((tags.get(258),tags.get(339,1)))
    if dtype is None:
        raise ValueError('Unsupported TIFF sample type.')
    return np.frombuffer(data,dtype=endian+dtype,count=tags[256]*tags[257],offset=tags[273]).reshape(tags[257],tags[256]).copy()


def surface_write(folder, array, ident=None):
    folder.mkdir(parents=True,exist_ok=True)
    arr=np.asarray(array,dtype='<f4')
    for i, name in enumerate(('z','y','x')):
        write_coordinate(folder/(name+'.tif'),arr[:,:,i])
    step_col=np.median(np.linalg.norm(np.diff(arr,axis=1),axis=2))
    step_row=np.median(np.linalg.norm(np.diff(arr,axis=0),axis=2))
    xyz=arr.reshape(-1,3)[:,::-1]
    save_json(folder/'meta.json',{'type':'seg','format':'tifxyz','uuid':ident or folder.stem,'name':ident or folder.stem,
                                   'scale':[1./float(step_col),1./float(step_row)],
                                   'bbox':[xyz.min(0).tolist(),xyz.max(0).tolist()]})


def surface_read(folder):
    coords=[read_tiff(folder/(name+'.tif')) for name in ('z','y','x')]
    if any(a.ndim!=2 or a.dtype.kind!='f' or a.dtype.itemsize!=4 for a in coords):
        raise ValueError('Expected three two-dimensional float32 coordinate rasters.')
    arr=np.stack(coords,axis=-1).astype('<f4',copy=False)
    # A partial sentinel is malformed. The working crop is in positive local coordinates.
    sentinel=arr==-1
    if np.any(sentinel.any(axis=-1) & ~sentinel.all(axis=-1)):
        raise ValueError('Inconsistent empty-vertex marker across TIFF channels.')
    valid=np.isfinite(arr).all(axis=-1) & (arr[:,:,0]>0) & ~sentinel.all(axis=-1)
    mask_file=folder/'mask.tif'
    if mask_file.exists():
        mask=read_tiff(mask_file)
        if mask.ndim==3:
            mask=mask[:,:,0]
        h,w=valid.shape
        if mask.ndim!=2 or mask.shape[0]%h or mask.shape[1]%w or mask.shape[0]<h or mask.shape[1]<w:
            raise ValueError('Niezgodny rozmiar maski.')
        sy,sx=mask.shape[0]//h,mask.shape[1]//w
        valid &= (mask>=255).reshape(h,sy,w,sx).all(axis=(1,3))
    return np.ascontiguousarray(arr),valid


def volume_write(folder, image):
    if image.ndim!=3 or image.dtype!=np.uint8:
        raise ValueError('Expected a uint8 CT in Z,Y,X order.')
    level=folder/'0'
    level.mkdir(parents=True,exist_ok=False)
    save_json(folder/'.zgroup',{'zarr_format':2})
    axes=[{'name':name,'type':'space'} for name in ('z','y','x')]
    data=[{'path':'0','coordinateTransformations':[{'type':'scale','scale':[1,1,1]}]}]
    save_json(folder/'.zattrs',{'multiscales':[{'version':'0.4','name':'ct','axes':axes,'datasets':data}]})
    depth,height,width=map(int,image.shape)
    save_json(folder/'meta.json',{'type':'vol','format':'zarr','name':'ct','uuid':'ct','width':width,
                                 'height':height,'slices':depth,'voxelsize':1.0,'min':0,'max':255})
    save_json(level/'.zarray',{'zarr_format':2,'shape':list(image.shape),'chunks':list(image.shape),
                              'dtype':'|u1','compressor':None,'fill_value':0,'order':'C','filters':None,
                              'dimension_separator':'.'})
    (level/'0.0.0').write_bytes(np.ascontiguousarray(image).tobytes())
