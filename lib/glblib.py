"""Minimal GLB reader/writer: pulls geometry accessors without decoding images."""
import json, struct, mmap
import numpy as np

CT = {5120:'i1',5121:'u1',5122:'i2',5123:'u2',5125:'u4',5126:'f4'}
NC = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}

class GLB:
    def __init__(self, path):
        self.path = path
        self.f = open(path,'rb')
        magic,ver,length = struct.unpack('<III', self.f.read(12))
        assert magic == 0x46546C67, "not a glb"
        clen,ctype = struct.unpack('<II', self.f.read(8))
        # Tolerant: accept a JSON chunk padded with NULs as well as the
        # spaces the spec asks for.
        self.js = json.loads(self.f.read(clen).rstrip(b'\x00 '))
        blen,btype = struct.unpack('<II', self.f.read(8))
        self.bin_off = 12+8+clen+8
        self.bin_len = blen
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def bv_bytes(self, i):
        bv = self.js['bufferViews'][i]
        o = self.bin_off + bv.get('byteOffset',0)
        return self.mm[o:o+bv['byteLength']]

    def acc(self, i):
        a = self.js['accessors'][i]
        n = a['count']; nc = NC[a['type']]; dt = np.dtype(CT[a['componentType']])
        bv = self.js['bufferViews'][a['bufferView']]
        base = self.bin_off + bv.get('byteOffset',0) + a.get('byteOffset',0)
        stride = bv.get('byteStride') or nc*dt.itemsize
        if stride == nc*dt.itemsize:
            arr = np.frombuffer(self.mm, dtype=dt, count=n*nc, offset=base)
            return arr.reshape(n,nc) if nc>1 else arr
        need = (n-1)*stride + nc*dt.itemsize
        raw = np.frombuffer(self.mm, dtype='u1', count=need, offset=base)
        pad = np.zeros(n*stride - need, 'u1')
        raw = np.concatenate([raw, pad]).reshape(n, stride)[:, :nc*dt.itemsize].copy()
        arr = raw.view(dt)
        return arr if nc>1 else arr.ravel()

    def close(self):
        self.mm.close(); self.f.close()


def pad4(b):
    r = len(b) % 4
    return b + (b'\x00' * (4-r) if r else b'')


class GLBWriter:
    """Builds a GLB from numpy arrays + raw image bytes."""
    def __init__(self):
        self.chunks = []      # list of bytes
        self.off = 0
        self.js = {"asset":{"version":"2.0","generator":"opencity-forge"},
                   "scene":0,"scenes":[{"nodes":[]}],"nodes":[],"meshes":[],
                   "accessors":[],"bufferViews":[],"materials":[],"images":[],
                   "samplers":[{"magFilter":9729,"minFilter":9987,
                                "wrapS":33071,"wrapT":33071},
                               {"magFilter":9729,"minFilter":9987,
                                "wrapS":10497,"wrapT":10497}],"textures":[]}

    def _bv(self, data, target=None):
        data = pad4(bytes(data))
        bv = {"buffer":0,"byteOffset":self.off,"byteLength":len(data)}
        if target: bv["target"]=target
        self.js['bufferViews'].append(bv)
        self.chunks.append(data); self.off += len(data)
        return len(self.js['bufferViews'])-1

    def _acc(self, arr, typ, ctype, target, minmax=False):
        bvi = self._bv(arr.tobytes(), target)
        a = {"bufferView":bvi,"componentType":ctype,"count":int(arr.shape[0]),
             "type":typ}
        if minmax:
            a["min"]=[float(x) for x in arr.min(axis=0)]
            a["max"]=[float(x) for x in arr.max(axis=0)]
        self.js['accessors'].append(a)
        return len(self.js['accessors'])-1

    def add_material_tex(self, texi, name=None):
        m = {"pbrMetallicRoughness":{
                "baseColorTexture":{"index":texi},
                "metallicFactor":0.0,"roughnessFactor":1.0},
             "doubleSided":True}
        if name: m["name"]=name
        self.js['materials'].append(m)
        return len(self.js['materials'])-1

    def add_material_vcol(self, name=None):
        m = {"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],
                "metallicFactor":0.0,"roughnessFactor":1.0},
             "doubleSided":True}
        if name: m["name"]=name
        self.js['materials'].append(m)
        return len(self.js['materials'])-1

    def add_node(self, name, pos, idx, uv=None, color=None, mat=None):
        pos = np.ascontiguousarray(pos, 'f4')
        attrs = {"POSITION": self._acc(pos,"VEC3",5126,34962,minmax=True)}
        if uv is not None:
            attrs["TEXCOORD_0"] = self._acc(np.ascontiguousarray(uv,'f4'),
                                            "VEC2",5126,34962)
        if color is not None:
            attrs["COLOR_0"] = self._acc(np.ascontiguousarray(color,'u1'),
                                         "VEC4",5121,34962)
            self.js['accessors'][attrs["COLOR_0"]]["normalized"]=True
        idx = np.ascontiguousarray(idx.ravel(),'u4')
        ia = self._acc(idx,"SCALAR",5125,34963)
        prim = {"attributes":attrs,"indices":ia,"mode":4}
        if mat is not None: prim["material"]=mat
        self.js['meshes'].append({"primitives":[prim],"name":name})
        self.js['nodes'].append({"mesh":len(self.js['meshes'])-1,"name":name})
        self.js['scenes'][0]['nodes'].append(len(self.js['nodes'])-1)

    def write(self, path, extras=None):
        if extras: self.js['extras']=extras
        for k in ("images","textures","materials"):
            if not self.js[k]: del self.js[k]
        if "images" not in self.js and "samplers" in self.js:
            del self.js['samplers']
        self.js['buffers']=[{"byteLength":self.off}]
        jb = json.dumps(self.js,separators=(',',':')).encode('utf8')
        jb += b' ' * ((4 - len(jb) % 4) % 4)   # glTF: JSON chunk pads with spaces
        total = 12 + 8 + len(jb) + 8 + self.off
        with open(path,'wb') as f:
            f.write(struct.pack('<III',0x46546C67,2,total))
            f.write(struct.pack('<II',len(jb),0x4E4F534A)); f.write(jb)
            f.write(struct.pack('<II',self.off,0x004E4942))
            for c in self.chunks: f.write(c)
        return total


class StreamGLBWriter:
    """Builds a GLB writing the binary chunk straight to disk.

    Peak memory is one buffer at a time, not the whole model, so large models
    do not need the whole binary chunk in RAM."""
    def __init__(self, binpath):
        self.binpath = binpath
        self.bf = open(binpath, 'wb')
        self.off = 0
        self.js = {"asset":{"version":"2.0","generator":"opencity-forge"},
                   "scene":0,"scenes":[{"nodes":[]}],"nodes":[],"meshes":[],
                   "accessors":[],"bufferViews":[],"materials":[],"images":[],
                   "samplers":[{"magFilter":9729,"minFilter":9987,
                                "wrapS":33071,"wrapT":33071},
                               {"magFilter":9729,"minFilter":9987,
                                "wrapS":10497,"wrapT":10497}],"textures":[]}

    def _write(self, data):
        self.bf.write(data)
        n = len(data)
        r = n % 4
        if r:
            self.bf.write(b"\x00"*(4-r)); n += 4-r
        return n

    def _bv_from_bytes(self, data, target=None):
        start = self.off
        n = self._write(data)
        bv = {"buffer":0,"byteOffset":start,"byteLength":len(data)}
        if target: bv["target"]=target
        self.js['bufferViews'].append(bv); self.off = start + n
        return len(self.js['bufferViews'])-1

    def _bv_from_file(self, path, target=None, chunk=1<<22):
        start = self.off; total = 0
        with open(path,'rb') as f:
            while True:
                b = f.read(chunk)
                if not b: break
                self.bf.write(b); total += len(b)
        r = total % 4
        if r: self.bf.write(b"\x00"*(4-r))
        bv = {"buffer":0,"byteOffset":start,"byteLength":total}
        if target: bv["target"]=target
        self.js['bufferViews'].append(bv)
        self.off = start + total + ((4-r) if r else 0)
        return len(self.js['bufferViews'])-1

    def _acc(self, arr, typ, ctype, target, minmax=False):
        import numpy as np
        arr = np.ascontiguousarray(arr)
        bvi = self._bv_from_bytes(arr.tobytes(), target)
        a = {"bufferView":bvi,"componentType":ctype,"count":int(arr.shape[0]),
             "type":typ}
        if minmax:
            a["min"]=[float(x) for x in arr.min(axis=0)]
            a["max"]=[float(x) for x in arr.max(axis=0)]
        self.js['accessors'].append(a)
        return len(self.js['accessors'])-1

    def add_image_file(self, path, mime="image/jpeg", wrap=0):
        """wrap=0 clamps (atlas crops), wrap=1 repeats (tiling materials)."""
        bvi = self._bv_from_file(path)
        self.js['images'].append({"bufferView":bvi,"mimeType":mime})
        ii = len(self.js['images'])-1
        self.js['textures'].append({"sampler":int(wrap),"source":ii})
        return len(self.js['textures'])-1

    def add_material_tex(self, texi, name=None):
        m = {"pbrMetallicRoughness":{"baseColorTexture":{"index":texi},
             "metallicFactor":0.0,"roughnessFactor":1.0},"doubleSided":True}
        if name: m["name"]=name
        self.js['materials'].append(m); return len(self.js['materials'])-1

    def add_material_vcol(self, name=None):
        m = {"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1],
             "metallicFactor":0.0,"roughnessFactor":1.0},"doubleSided":True}
        if name: m["name"]=name
        self.js['materials'].append(m); return len(self.js['materials'])-1

    def add_node(self, name, pos, idx, uv=None, color=None, mat=None,
                 extras=None, normal=None):
        import numpy as np
        pos = np.ascontiguousarray(pos,'f4')
        attrs = {"POSITION": self._acc(pos,"VEC3",5126,34962,minmax=True)}
        if normal is not None:
            attrs["NORMAL"]=self._acc(np.ascontiguousarray(normal,'f4'),
                                      "VEC3",5126,34962)
        if uv is not None:
            attrs["TEXCOORD_0"]=self._acc(np.ascontiguousarray(uv,'f4'),
                                          "VEC2",5126,34962)
        if color is not None:
            ci=self._acc(np.ascontiguousarray(color,'u1'),"VEC4",5121,34962)
            self.js['accessors'][ci]["normalized"]=True
            attrs["COLOR_0"]=ci
        ia=self._acc(np.ascontiguousarray(idx.ravel(),'u4'),"SCALAR",5125,34963)
        prim={"attributes":attrs,"indices":ia,"mode":4}
        if mat is not None: prim["material"]=mat
        mesh={"primitives":[prim],"name":name}
        self.js['meshes'].append(mesh)
        node={"mesh":len(self.js['meshes'])-1,"name":name}
        if extras: node["extras"]=extras
        self.js['nodes'].append(node)
        self.js['scenes'][0]['nodes'].append(len(self.js['nodes'])-1)

    def write(self, path, extras=None, chunk=1<<22):
        import struct, json as _json, os
        self.bf.flush(); self.bf.close()
        if extras: self.js['extras']=extras
        for k in ("images","textures","materials"):
            if not self.js.get(k): self.js.pop(k,None)
        if "images" not in self.js: self.js.pop('samplers',None)
        self.js['buffers']=[{"byteLength":self.off}]
        jb=_json.dumps(self.js,separators=(',',':')).encode('utf8')
        r=len(jb)%4
        if r: jb+=b" "*(4-r)
        total=12+8+len(jb)+8+self.off
        with open(path,'wb') as f:
            f.write(struct.pack('<III',0x46546C67,2,total))
            f.write(struct.pack('<II',len(jb),0x4E4F534A)); f.write(jb)
            f.write(struct.pack('<II',self.off,0x004E4942))
            with open(self.binpath,'rb') as b:
                while True:
                    d=b.read(chunk)
                    if not d: break
                    f.write(d)
        os.unlink(self.binpath)
        return total


# --- scene-graph extensions for the model build ------------------------ #
def _sg_add_mesh(self, name, pos, idx, uv=None, color=None, mat=None, normal=None):
    """Register a mesh and return its index without attaching it to a node."""
    import numpy as np
    pos = np.ascontiguousarray(pos,'f4')
    attrs = {"POSITION": self._acc(pos,"VEC3",5126,34962,minmax=True)}
    if normal is not None:
        attrs["NORMAL"]=self._acc(np.ascontiguousarray(normal,'f4'),"VEC3",5126,34962)
    if uv is not None:
        attrs["TEXCOORD_0"]=self._acc(np.ascontiguousarray(uv,'f4'),"VEC2",5126,34962)
    if color is not None:
        ci=self._acc(np.ascontiguousarray(color,'u1'),"VEC4",5121,34962)
        self.js['accessors'][ci]["normalized"]=True
        attrs["COLOR_0"]=ci
    ia=self._acc(np.ascontiguousarray(idx.ravel(),'u4'),"SCALAR",5125,34963)
    prim={"attributes":attrs,"indices":ia,"mode":4}
    if mat is not None: prim["material"]=mat
    self.js['meshes'].append({"primitives":[prim],"name":name})
    return len(self.js['meshes'])-1

def _sg_node(self, name, mesh=None, children=None, translation=None,
             scale=None, extras=None, root=False):
    """Create a node.  root=True puts it in the scene; otherwise it is expected
    to be someone's child."""
    n = {"name": name}
    if mesh is not None: n["mesh"]=mesh
    if children: n["children"]=list(children)
    if translation is not None: n["translation"]=[float(v) for v in translation]
    if scale is not None: n["scale"]=[float(v) for v in scale]
    if extras: n["extras"]=extras
    self.js['nodes'].append(n)
    i = len(self.js['nodes'])-1
    if root: self.js['scenes'][0]['nodes'].append(i)
    return i

def _sg_add_child_node(self, name, pos, idx, uv=None, color=None, mat=None,
                       extras=None, normal=None):
    """add_node, but returns the node index and does NOT add it to the scene
    root, so it can be parented under a layer group."""
    m = self.add_mesh(name, pos, idx, uv=uv, color=color, mat=mat, normal=normal)
    return self.node(name, mesh=m, extras=extras)

def _sg_add_material_plain(self, rgba, name=None, rough=1.0, metal=0.0):
    m={"pbrMetallicRoughness":{"baseColorFactor":[float(v) for v in rgba],
       "metallicFactor":float(metal),"roughnessFactor":float(rough)},
       "doubleSided":True}
    if name: m["name"]=name
    if len(rgba)>3 and rgba[3]<1.0: m["alphaMode"]="BLEND"
    self.js['materials'].append(m); return len(self.js['materials'])-1

for _cls in (GLBWriter, StreamGLBWriter):
    _cls.add_mesh = _sg_add_mesh
    _cls.node = _sg_node
    _cls.child_node = _sg_add_child_node
    _cls.add_material_plain = _sg_add_material_plain
