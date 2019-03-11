#! /usr/bin/env python2

"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""

# Requirements: Numpy as PIL/Pillow
import numpy as np
try:
    import png
    has_png = True
except:
    has_png = False
    png=None



# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'.encode()

def flow_read(filename, return_validity=False):
    """ Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]

    if return_validity:
        valid = u<1e19
        u[valid==0] = 0
        v[valid==0] = 0
        return u,v,valid
    else:
        return u,v

def flow_write(filename,uv,v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        uv_ = np.array(uv)
        assert(uv_.ndim==3)
        if uv_.shape[0] == 2:
            u = uv_[0,:,:]
            v = uv_[1,:,:]
        elif uv_.shape[2] == 2:
            u = uv_[:,:,0]
            v = uv_[:,:,1]
        else:
            raise UVError('Wrong format for flow input')
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def flow_read_png(fpath):
    """
    Read KITTI optical flow, returns u,v,valid mask

    """
    if not has_png:
        print('Error. Please install the PyPNG library')
        return

    R = png.Reader(fpath)
    width,height,data,_ = R.asDirect()
    # This only worked with python2.
    #I = np.array(map(lambda x:x,data)).reshape((height,width,3))
    I = np.array([x for x in data]).reshape((height,width,3))
    u_ = I[:,:,0]
    v_ = I[:,:,1]
    valid = I[:,:,2]

    u = (u_.astype('float64')-2**15)/64.0
    v = (v_.astype('float64')-2**15)/64.0

    return u,v,valid


def flow_write_png(fpath,u,v,valid=None):
    """
    Write KITTI optical flow.

    """
    if not has_png:
        print('Error. Please install the PyPNG library')
        return


    if valid==None:
        valid_ = np.ones(u.shape,dtype='uint16')
    else:
        valid_ = valid.astype('uint16')


    u = u.astype('float64')
    v = v.astype('float64')

    u_ = ((u*64.0)+2**15).astype('uint16')
    v_ = ((v*64.0)+2**15).astype('uint16')

    I = np.dstack((u_,v_,valid_))

    W = png.Writer(width=u.shape[1],
                   height=u.shape[0],
                   bitdepth=16,
                   planes=3)

    with open(fpath,'wb') as fil:
        W.write(fil,I.reshape((-1,3*u.shape[1])))
