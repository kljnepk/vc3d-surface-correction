#!/usr/bin/env python3
"""Prepare a frozen R094 example from an authorized source TIFF; never downloads data."""
import argparse, hashlib, json, sys, tempfile, struct
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from r103.formats import surface_read, surface_write, volume_write

def digest(b):return hashlib.sha256(b).hexdigest()
def file_digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def case_info(case):
    return next(e for e in json.loads((HERE/'examples.json').read_text())['cases'] if e['case']==case)
def frozen(e):
    meshes={a:np.asarray(m,dtype='<f4') for a,m in e['meshes_zyx'].items()}
    for a,m in meshes.items():
        require(m.shape==(33,9,3) and np.isfinite(m).all() and digest(m.tobytes())==e['mesh_sha256'][a],f'{a} frozen coordinate digest mismatch')
    require(all(np.array_equal(m[:3],meshes['VC3D'][:3]) for m in meshes.values()),'Starting edges differ')
    return meshes

def normalize(raw):
    require(raw.shape==(128,128,128),'Expected exact 128-cubed crop')
    lo,hi=np.percentile(raw[::3,::3,::3],[2,98]);require(hi>lo,'Constant crop')
    return np.rint(255*np.clip((raw.astype(float)-lo)/(hi-lo),0,1)).astype('uint8')

def write_example(e,ct,out):
    require(ct.dtype==np.uint8 and ct.shape==(128,128,128) and digest(ct.tobytes())==e['normalized_crop_sha256'],'Normalized crop digest mismatch')
    meshes=frozen(e);out=out.resolve();require(not out.exists(),'Choose a new output directory');out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage=Path(tmp)/'stage';stage.mkdir();volume_write(stage/'ct.zarr',ct);surface_write(stage/'seed.tifxyz',meshes['VC3D'][:3]);
        for a,m in meshes.items():
            p=stage/'frozen'/(a+'.tifxyz');surface_write(p,m);actual,valid=surface_read(p);require(valid.all() and np.array_equal(actual,m),f'{a} write/read mismatch')
        (stage/'provenance.json').write_text(json.dumps({k:v for k,v in e.items() if k!='meshes_zyx'},indent=2)+'\n');stage.rename(out)
    print('PREPARE=PASS_EXACT_CROP_AND_FROZEN_COORDINATES',out)

def raw_tiff_crop(path, origin, side):
    """Uncompressed grayscale classic multi-page TIFF fallback; no dependencies except NumPy."""
    import numpy as np
    z0,y0,x0=origin
    with path.open('rb') as f:
        def read_at(pos,n):
            f.seek(pos); b=f.read(n); require(len(b)==n,'Truncated TIFF'); return b
        head=read_at(0,8); order={b'II':'<',b'MM':'>'}.get(head[:2])
        require(order and struct.unpack_from(order+'H',head,2)[0]==42,'This TIFF requires tifffile or Pillow')
        pos=struct.unpack_from(order+'I',head,4)[0]; seen=set();frames=[]
        for index in range(side):
            require(pos and pos not in seen,'Invalid TIFF page chain');seen.add(pos)
            count=struct.unpack(order+'H',read_at(pos,2))[0]; directory=read_at(pos+2,12*count+4)
            pos=struct.unpack_from(order+'I',directory,12*count)[0]
            if not z0<=index<z0+128: continue
            tags={}
            for j in range(count):
                off=j*12; tag,kind,n=struct.unpack_from(order+'HHI',directory,off)
                if tag not in (256,257,258,259,262,273,277,279,339) or kind not in (3,4): continue
                require(n<=65536,'TIFF tag table is too long'); width,code=(2,'H') if kind==3 else (4,'I');length=n*width
                raw=directory[off+8:off+8+length] if length<=4 else read_at(struct.unpack_from(order+'I',directory,off+8)[0],length)
                tags[tag]=struct.unpack(order+str(n)+code,raw)
            require({256,257,258,273,279}.issubset(tags),'Missing TIFF strip metadata')
            require((tags[256][0],tags[257][0])==(side,side) and tags.get(259,(1,))[0]==1 and tags.get(277,(1,))[0]==1 and tags.get(339,(1,))[0]==1 and tags.get(262,(1,))[0]==1 and tags[258][0] in (8,16),'This TIFF requires tifffile or Pillow')
            size=tags[258][0]//8
            require(len(tags[273])==len(tags[279]) and sum(tags[279])==side*side*size,'Invalid TIFF strips')
            raw=b''.join(read_at(s,n) for s,n in zip(tags[273],tags[279]))
            frame=np.frombuffer(raw,dtype=order+'u'+str(size)).reshape(side,side)
            frames.append(frame[y0:y0+128,x0:x0+128].copy())
        require(not pos,'Too many TIFF pages')
    return np.stack(frames)

def read_crop(path,origin,side):
    import numpy as np
    z0,y0,x0=origin; ys=slice(y0,y0+128);xs=slice(x0,x0+128)
    errors=[]
    try:
        from tifffile import TiffFile
        with TiffFile(path) as t:
            if len(t.pages)==side and t.pages[0].shape==(side,side):
                arr=np.stack([t.pages[i].asarray()[ys,xs] for i in range(z0,z0+128)])
            else:
                a=t.asarray();require(a.shape==(side,)*3,'Unexpected TIFF dimensions');arr=a[z0:z0+128,ys,xs].copy()
        require(arr.shape==(128,)*3 and arr.dtype.kind in 'bui','Unexpected CT dtype or shape');return arr,'tifffile'
    except Exception as e: errors.append(str(e))
    try:
        from PIL import Image
        with Image.open(path) as im:
            require(im.size==(side,side) and im.n_frames==side,'Unexpected TIFF dimensions')
            frames=[]
            for i in range(z0,z0+128): im.seek(i);frames.append(np.array(im.crop((x0,y0,x0+128,y0+128))))
        arr=np.stack(frames);require(arr.shape==(128,)*3 and arr.dtype.kind in 'bui','Unexpected CT dtype or shape');return arr,'Pillow'
    except Exception as e: errors.append(str(e))
    try: return raw_tiff_crop(path,origin,side),'classic TIFF'
    except Exception as e: raise ValueError('Unable to read CT: '+' | '.join(errors+[str(e)])) from e


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,choices=['01','02','03','04','05']);p.add_argument('--volume',type=Path);p.add_argument('--out',type=Path);p.add_argument('--check-parents',type=Path);a=p.parse_args();e=case_info(a.case)
    if a.check_parents is not None:
        if a.volume or a.out:p.error('--check-parents cannot be combined with --volume or --out')
        frozen(e)
        ok=True
        for arm in ('VC3D','R094'):
            mesh,valid=surface_read(a.check_parents/(arm+'.tifxyz'));h=digest(mesh.astype('<f4',copy=False).tobytes());match=valid.all() and h==e['mesh_sha256'][arm];ok &= bool(match);print(arm,'EXACT_MATCH' if match else 'MISMATCH',h)
        return 0 if ok else 2
    if a.volume is None or a.out is None:p.error('Provide --volume and --out, or --check-parents')
    require(a.volume.is_file() and a.volume.stat().st_size==e['source']['bytes'] and file_digest(a.volume)==e['source']['sha256'],'Source TIFF digest/size mismatch')
    side=int(e['source']['path'].removesuffix('_0000.tif').rsplit('_',1)[-1]); raw,reader=read_crop(a.volume,e['crop_origin_zyx'],side);ct=normalize(raw);write_example(e,ct,a.out);print('TIFF_READER='+reader);return 0
if __name__=='__main__':raise SystemExit(main())
