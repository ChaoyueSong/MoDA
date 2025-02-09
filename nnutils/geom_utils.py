# Modified from https://github.com/facebookresearch/banmo

import pdb
import time
import cv2
import numpy as np
import trimesh
from pytorch3d import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, 'third_party')
from ext_utils.flowlib import warp_flow, cat_imgflo 
from dual_quat import q_normalize, q_mul, dq_mul, dq_normalize, dq_quaternion_conjugate, dq_combined_conjugate, dq_inverse

def evaluate_mlp(model, xyz_embedded, embed_xyz=None, dir_embedded=None,
                chunk=32*1024, 
                xyz=None,
                code=None, appearance_code=None, sigma_only=False, use_semantic=False):
    """
    embed_xyz: embedding function
    chunk is the point-level chunk divided by number of bins
    """
    B,nbins,k = xyz_embedded.shape
 
    out_chunks = []
    for i in range(0, B, chunk):
        embedded = xyz_embedded[i:i+chunk]
        if embed_xyz is not None:
            embedded = embed_xyz(embedded)
        if dir_embedded is not None:
            embedded = torch.cat([embedded,
                       dir_embedded[i:i+chunk]], -1)
        if code is not None:
            if code.shape[0] != B and code.dim() == 2:
                code = code.repeat(B,1)
            code_chunk = code[i:i+chunk]
            if code_chunk.dim() == 2: 
                code_chunk = code_chunk[:,None]
            code_chunk = code_chunk.repeat(1,nbins,1)
            embedded = torch.cat([embedded,code_chunk], -1)
        if appearance_code is not None:  #add appearance_code for nerf_coarse
            appearance_code_chunk = appearance_code[i:i+chunk]
            if appearance_code_chunk.dim() == 2: 
                appearance_code_chunk = appearance_code_chunk[:,None]
            appearance_code_chunk = appearance_code_chunk.repeat(1,nbins,1)
            embedded = torch.cat([embedded,appearance_code_chunk], -1)
        if xyz is not None:
            xyz_chunk = xyz[i:i+chunk]
        else: xyz_chunk = None
        out_chunks += [model(embedded, sigma_only=sigma_only, xyz=xyz_chunk)]

    out = torch.cat(out_chunks, 0)
    return out

def bone_transform(bones_in, rts, neudbs, is_vec=False):
    """ 
    bones_in: 1,B,10  - B gaussian ellipsoids of bone coordinates
    rts: ...,B,3,4    - B ririd transforms
    new_rts: ...,8
    rts are applied to bone coordinate transforms (left multiply)
    is_vec:     whether rts are stored as r1...9,t1...3 vector form
    """
    B = bones_in.shape[-2]
    bones = bones_in.view(-1,B,10).clone()
    center = bones[:,:,:3]
    orient = bones[:,:,3:7] # real first
    scale =  bones[:,:,7:10]

    if neudbs:
        rts = rts.view(-1,B,8)
        bs = rts.shape[0] 

        dq_r = rts[..., :4]
        dq_d = rts[..., 4:]
        
        Rmat = transforms.quaternion_to_matrix(dq_r)
        Tmat = 2 * transforms.quaternion_raw_multiply(dq_d, transforms.quaternion_invert(dq_r))
        Tmat = Tmat[...,1:]

        # move bone coordinates (left multiply)
        center = Rmat.matmul(center[...,None])[...,0]+Tmat
        orient = transforms.quaternion_multiply(dq_r, orient)
    else: # use lbs
        if is_vec:
            rts = rts.view(-1,B,12)
        else:
            rts = rts.view(-1,B,3,4)
        bs = rts.shape[0] 

        center = bones[:,:,:3]
        orient = bones[:,:,3:7] # real first
        scale =  bones[:,:,7:10]
        if is_vec:
            Rmat = rts[:,:,:9].view(-1,B,3,3)
            Tmat = rts[:,:,9:12].view(-1,B,3,1)
        else:
            Rmat = rts[:,:,:3,:3]   
            Tmat = rts[:,:,:3,3:4]   

        # move bone coordinates (left multiply)
        center = Rmat.matmul(center[...,None])[...,0]+Tmat[...,0]
        Rquat = transforms.matrix_to_quaternion(Rmat)
        orient = transforms.quaternion_multiply(Rquat, orient)

    scale = scale.repeat(bs,1,1)
    bones = torch.cat([center,orient,scale],-1)
    return bones 

def rtmat_invert(Rmat, Tmat):
    """
    Rmat: ...,3,3   - rotations
    Tmat: ...,3   - translations
    """
    rts = torch.cat([Rmat, Tmat[...,None]],-1)
    rts_i = rts_invert(rts)
    Rmat_i = rts_i[...,:3,:3] # bs, B, 3,3
    Tmat_i = rts_i[...,:3,3]
    return Rmat_i, Tmat_i

def rtk_invert(rtk_in, B):
    """
    rtk_in: ... (rot 1...9, trans 1...3)
    """
    rtk_shape = rtk_in.shape
    rtk_in = rtk_in.view(-1,B,12)# B,12
    rmat=rtk_in[:,:,:9]
    rmat=rmat.view(-1,B,3,3)
    tmat= rtk_in[:,:,9:12]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)
    rts_bw = rts_invert(rts_fw)

    rvec = rts_bw[...,:3,:3].reshape(-1,9)
    tvec = rts_bw[...,:3,3] .reshape(-1,3)
    rtk = torch.cat([rvec,tvec],-1).view(rtk_shape)
    return rtk

def rts_invert(rts_in):
    """
    rts: ...,3,4   - B ririd transforms
    """
    rts = rts_in.view(-1,3,4).clone()
    Rmat = rts[:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:3,3:]
    Rmat_i=Rmat.permute(0,2,1)
    Tmat_i=-Rmat_i.matmul(Tmat)
    rts_i = torch.cat([Rmat_i, Tmat_i],-1)
    rts_i = rts_i.view(rts_in.shape)
    return rts_i

def rtk_to_4x4(rtk):
    """
    rtk: ...,12
    """
    device = rtk.device
    bs = rtk.shape[0]
    zero_one = torch.Tensor([[0,0,0,1]]).to(device).repeat(bs,1)

    rmat=rtk[:,:9]
    rmat=rmat.view(-1,3,3)
    tmat=rtk[:,9:12]
    rts = torch.cat([rmat,tmat[...,None]],-1)
    rts = torch.cat([rts,zero_one[:,None]],1)
    return rts

def rtk_compose(rtk1, rtk2):
    """
    rtk ...
    """
    rtk_shape = rtk1.shape
    rtk1 = rtk1.view(-1,12)# ...,12
    rtk2 = rtk2.view(-1,12)# ...,12

    rts1 = rtk_to_4x4(rtk1)
    rts2 = rtk_to_4x4(rtk2)

    rts = rts1.matmul(rts2)
    rvec = rts[...,:3,:3].reshape(-1,9)
    tvec = rts[...,:3,3].reshape(-1,3)
    rtk = torch.cat([rvec,tvec],-1).view(rtk_shape)
    return rtk

def vec_to_sim3(vec):
    """
    vec:      ...,10
    center:   ...,3
    orient:   ...,3,3
    scale:    ...,3
    """
    center = vec[...,:3]
    orient = vec[...,3:7] # real first
    orient = F.normalize(orient, 2,-1)
    orient = transforms.quaternion_to_matrix(orient) # real first
    scale =  vec[...,7:10].exp()
    return center, orient, scale


def gauss_mlp_skinning(xyz, embedding_xyz, bones, 
                    pose_code,  nerf_skin, skin_aux=None):
    """
    xyz:        N_rays, ndepth, 3
    bones:      ... nbones, 10
    pose_code:  ...,1, nchannel
    """
    N_rays = xyz.shape[0]
    #TODO hacky way to make code compaitible with noqueryfw
    if pose_code.dim() == 2 and pose_code.shape[0]!=N_rays: 
        pose_code = pose_code[None].repeat(N_rays, 1,1)

    xyz_embedded = embedding_xyz(xyz)
    dskin = mlp_skinning(nerf_skin, pose_code, xyz_embedded)    #delta(W)
    skin = skinning(bones, xyz, dskin, skin_aux=skin_aux) # bs, N, B  : skinning weight: W
    return skin

def mlp_skinning(mlp, code, pts_embed):
    """
    code: bs, D          - N D-dimensional pose code
    pts_embed: bs,N,x    - N point positional embeddings
    dskin: bs,N,B        - delta skinning matrix
    """
    if mlp is None:
        dskin = None
    else:
        dskin = evaluate_mlp(mlp, pts_embed, code=code, chunk=8*1024)
    return dskin

def axis_rotate(orient, mdis):
    bs,N,B,_,_ = mdis.shape
    mdis = (orient * mdis.view(bs,N,B,1,3)).sum(4)[...,None] # faster 
    #mdis = orient.matmul(mdis) # bs,N,B,3,1 # slower
    return mdis

def skinning_chunk(bones, pts, dskin=None, skin_aux=None):
    """
    bone: bs,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points, usually N=num points per ray, b~=2034
    skin: bs,N,B   - skinning matrix
    """
    device = pts.device
    log_scale= skin_aux[0]
    w_const  = skin_aux[1]
    bs,N,_ = pts.shape
    B = bones.shape[-2]
    if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
    bones = bones.view(-1,B,10)
   
    center, orient, scale = vec_to_sim3(bones) 
    orient = orient.permute(0,1,3,2) # transpose R

    # mahalanobis distance [(p-v)^TR^T]S[R(p-v)]
    # transform a vector to the local coordinate
    mdis = center.view(bs,1,B,3) - pts.view(bs,N,1,3) # bs,N,B,3    #Equation 11 in banmo
    if True:#B<50:
        mdis = axis_rotate(orient.view(bs,1,B,3,3), mdis[...,None])
        #mdis = orient.view(bs,1,B,3,3).matmul(mdis[...,None]) # bs,N,B,3,1
        mdis = mdis[...,0]
        mdis = scale.view(bs,1,B,3) * mdis.pow(2)
    else:
        # for efficiency considerations
        mdis = mdis.pow(2)
    mdis = mdis*100*log_scale.exp() # TODO accound for scaled near-far plane
    mdis = (-10 * mdis.sum(3)) # bs,N,B

    if dskin is not None:
        mdis = mdis+dskin    #W+deltaW  equation 12

    ## truncated softmax
    #max_bone=min(B,3)
    #topk, indices = mdis.topk(max_bone, 2, largest=True)
    #mdis = torch.zeros_like(mdis).fill_(-np.inf)
    #mdis = mdis.scatter(2, indices, topk)
    skin = mdis.softmax(2)
    return skin
    

def skinning(bones, pts, dskin=None, skin_aux=None):
    """
    bone: ...,B,10  - B gaussian ellipsoids
    pts: bs,N,3    - N 3d points
    skin: bs,N,B   - skinning matrix
    """
    chunk=4096
    bs,N,_ = pts.shape
    B = bones.shape[-2]
    if bones.dim()==2: bones = bones[None].repeat(bs,1,1)
    bones = bones.view(-1,B,10)

    skin = []
    for i in range(0,bs,chunk):
        if dskin is None:
            dskin_chunk = None
        else: 
            dskin_chunk = dskin[i:i+chunk]
        skin_chunk = skinning_chunk(bones[i:i+chunk], pts[i:i+chunk], \
                              dskin=dskin_chunk, skin_aux=skin_aux)
        skin.append( skin_chunk )
    skin = torch.cat(skin,0)
    return skin

def blend_skinning_chunk(rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    B = rts.shape[-3]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    Rmat = rts[:,:,:3,:3] # bs, B, 3,3
    Tmat = rts[:,:,:3,3]
    device = Tmat.device

    # Gi=sum(wbGb), V=RV+T
    Rmat_w = (skin[...,None,None] * Rmat[:,None]).sum(2) # bs,N,B,3,3-->bs,N,3,3
    Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3-->bs,N,3
    pts = Rmat_w.matmul(pts[...,None]) + Tmat_w[...,None] 
    pts = pts[...,0]
    return pts

def blend_skinning(rts, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    chunk=4096
    B = rts.shape[-3]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    rts = rts.view(-1,B,3,4)
    bs = pts.shape[0]

    pts_out = []
    for i in range(0,bs,chunk):
        pts_chunk = blend_skinning_chunk(rts[i:i+chunk], 
                                          skin[i:i+chunk], pts[i:i+chunk])
        pts_out.append(pts_chunk)
    pts = torch.cat(pts_out,0)
    return pts

def calculate_residual_deformation(nerf_dis, xyz, embedding_xyz, code):
    """
    calculated displacement field for non-rigid deformation
    """

    xyz_dis = evaluate_mlp(nerf_dis, xyz, embedding_xyz, code=code, chunk=xyz.shape[0])  # chunk same as nerf_skin, not sure, have bug in about 90th epoch.
    # pts = embedding(tpose)
    # pts = pts.transpose(1, 2)
    # latent = batch['poses']
    # latent = latent[..., None].expand(*latent.shape, pts.size(2))
    # features = torch.cat((pts, latent), dim=1)

    # net = features
    # for i, l in enumerate(self.resd_linears):
    #     net = self.actvn(self.resd_linears[i](net))
    #     if i in self.skips:
    #         net = torch.cat((features, net), dim=1)
    # resd = self.resd_fc(net)
    # resd = resd.transpose(1, 2)
    # resd = 0.05 * torch.tanh(resd)
    return xyz_dis

def neu_dbs(bones, rts_fw, skin, xyz_in, nerf_dis=None, embedding_xyz=None, code=None, backward=True):
    """
    bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
    rts_fw: bs,B,12       - B rigid transforms, applied to the rest bones
    xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
    nerf_dis, embedding_xyz, code for displacement field; code is pose_code/rest_pose_code
    """
    B = bones.shape[-2]
    N = xyz_in.shape[-2]
    bs = rts_fw.shape[0]
    bones = bones.view(-1,B,10)
    xyz_in = xyz_in.view(-1,N,3)
    rts_fw = rts_fw.view(-1,B,8)

    if backward:
        bones_dfm = bone_transform(bones, rts_fw, neudbs=True) # bone coordinates after deform
        dq = dq_inverse(rts_fw)   #rts_invert(rts_fw)
        
        xyz = dqs_blend_skinning(dq, skin, xyz_in)

        #xyz = direct_qs_blend_skinning(tmat, R_quat, skin, xyz_in)

        # weighted dq and normalized it
        # weighted_dq = (skin[...,None] * dq[:,None]).sum(2)   # (bs,N,B,1) * (bs,1,B,8)  = bs,N,B,8 -->bs,N,8
        # dq_normalized = dq_normalize(weighted_dq)

        # non_dual_dq = dq_normalized[..., :4]
        # dual_dq = dq_normalized[..., 4:]
        # scalar_non_dual = non_dual_dq[..., 0] #a0
        # vec_non_dual = non_dual_dq[..., 1:]  #d0
        # scalar_dual = dual_dq[..., 0] #aε
        # vec_dual = dual_dq[..., 1:]  #dε
        
        # trans = 2 * (scalar_non_dual[...,None] * vec_dual - scalar_dual[...,None] * vec_non_dual + torch.cross(vec_non_dual, vec_dual, dim=-1))
        # rotated_xyz = xyz_in + 2 * torch.cross(vec_non_dual, torch.cross(vec_non_dual, xyz_in, dim=-1) + scalar_non_dual[...,None] * xyz_in, dim=-1)
        # xyz = rotated_xyz + trans  
        
        # # apply to xyz_in, convert xyz_in to xyz_as_dq first
        # real_parts = xyz_in.new_zeros(xyz_in.shape[:-1] + (1,))
        # xyz_as_quaternion = torch.cat((real_parts, xyz_in), -1) 
        # xyz_as_dq_r = xyz_as_quaternion.new_zeros(xyz_as_quaternion.shape)
        # xyz_as_dq_r[...,0] = 1
        # xyz_as_dq = torch.cat((xyz_as_dq_r, xyz_as_quaternion), -1)  #bs,N,8
        # xyz = dq_mul(dq_mul(dq_normalized, xyz_as_dq), dq_combined_conjugate(dq_normalized))[..., 5:]
        if nerf_dis is not None:
            xyz_dis = calculate_residual_deformation(nerf_dis, xyz_in, embedding_xyz, code)
            xyz = xyz - xyz_dis 
    else:
        if nerf_dis is not None:
            xyz_dis = calculate_residual_deformation(nerf_dis, xyz_in, embedding_xyz, code)
            xyz_in = xyz_in + xyz_dis
        
        dq = rts_fw
        xyz = dqs_blend_skinning(dq, skin, xyz_in)

        #xyz = direct_qs_blend_skinning(tmat, R_quat, skin, xyz_in)

        # weighted dq and normalized it
        # weighted_dq = (skin[...,None] * dq[:,None]).sum(2)   # (bs,B,8) * (bs,N,B) = bs,N,B,8 
        # dq_normalized = dq_normalize(weighted_dq)

        # non_dual_dq = dq_normalized[..., :4]
        # dual_dq = dq_normalized[..., 4:]
        # scalar_non_dual = non_dual_dq[..., 0] #a0
        # vec_non_dual = non_dual_dq[..., 1:]  #d0
        # scalar_dual = dual_dq[..., 0] #aε
        # vec_dual = dual_dq[..., 1:]  #dε
        
        # trans = 2 * (scalar_non_dual[...,None] * vec_dual - scalar_dual[...,None] * vec_non_dual + torch.cross(vec_non_dual, vec_dual, dim=-1))
        # rotated_xyz = xyz_in + 2 * torch.cross(vec_non_dual, torch.cross(vec_non_dual, xyz_in, dim=-1) + scalar_non_dual[...,None] * xyz_in, dim=-1)
        # xyz = rotated_xyz + trans  

        # # apply to xyz_in, convert xyz_in to xyz_as_dq first
        # real_parts = xyz_in.new_zeros(xyz_in.shape[:-1] + (1,))
        # xyz_as_quaternion = torch.cat((real_parts, xyz_in), -1) 
        # xyz_as_dq_r = xyz_as_quaternion.new_zeros(xyz_as_quaternion.shape)
        # xyz_as_dq_r[...,0] = 1
        # xyz_as_dq = torch.cat((xyz_as_dq_r, xyz_as_quaternion), -1)
        # xyz = dq_mul(dq_mul(dq_normalized, xyz_as_dq), dq_combined_conjugate(dq_normalized))[..., 5:]

        bones_dfm = bone_transform(bones, rts_fw, neudbs=True) # bone coordinates after deform
    if nerf_dis is not None:
        return xyz, bones_dfm, xyz_dis
    else:
        return xyz, bones_dfm, 0
def dqs_blend_skinning_chunk(dq, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    B = dq.shape[-2]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)

    # weighted dq and normalized it
    weighted_dq = (skin[...,None] * dq[:,None]).sum(2)   # (bs,N,B,1) * (bs,1,B,8)  = bs,N,B,8 -->bs,N,8  b = b0 + ε bε
    dq_normalized = dq_normalize(weighted_dq)
    
    # apply to xyz_in, convert xyz_in to xyz_as_dq first
    # real_parts = pts.new_zeros(pts.shape[:-1] + (1,))
    # xyz_as_quaternion = torch.cat((real_parts, pts), -1) 
    # xyz_as_dq_r = xyz_as_quaternion.new_zeros(xyz_as_quaternion.shape)
    # xyz_as_dq_r[...,0] = 1
    # xyz_as_dq = torch.cat((xyz_as_dq_r, xyz_as_quaternion), -1)  #bs,N,8
    # xyz = dq_mul(dq_mul(dq_normalized, xyz_as_dq), dq_combined_conjugate(dq_normalized))[..., 5:]

    #do as paper: v'=v+2d0×(d0×v+a0v)+2(a0dε−aεd0+d0×dε), c0 = b0/||b0||=a0+d0, cε = bε/||b0||=aε+dε, x is cross product, a is scalar, d is vector
    non_dual_dq = dq_normalized[..., :4]
    dual_dq = dq_normalized[..., 4:]
    scalar_non_dual = non_dual_dq[..., 0] #a0
    vec_non_dual = non_dual_dq[..., 1:]  #d0
    scalar_dual = dual_dq[..., 0] #aε
    vec_dual = dual_dq[..., 1:]  #dε
    
    trans = 2 * (scalar_non_dual[...,None] * vec_dual - scalar_dual[...,None] * vec_non_dual + torch.cross(vec_non_dual, vec_dual, dim=-1))
    rotated_xyz = pts + 2 * torch.cross(vec_non_dual, torch.cross(vec_non_dual, pts, dim=-1) + scalar_non_dual[...,None] * pts, dim=-1)
    xyz = rotated_xyz + trans  

    return xyz

def dqs_blend_skinning(dq, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    chunk=4096
    B = dq.shape[-2]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    # rts = rts.view(-1,B,3,4)
    bs = pts.shape[0]
    dq = dq.view(-1,B,8)      #bs,B,8

    pts_out = []
    for i in range(0,bs,chunk):
        pts_chunk = dqs_blend_skinning_chunk(dq[i:i+chunk], 
                                          skin[i:i+chunk], pts[i:i+chunk])
        pts_out.append(pts_chunk)
    pts = torch.cat(pts_out,0)
    return pts

def direct_qs_blend_skinning(Tmat, q_r, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    chunk=4096
    B = dq.shape[-2]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)
    # rts = rts.view(-1,B,3,4)
    bs = pts.shape[0]
    q_r = q_r.view(-1,B,4)      #bs,B,8

    pts_out = []
    for i in range(0,bs,chunk):
        pts_chunk = direct_qs_blend_skinning_chunk(Tmat[i:i+chunk], q_r[i:i+chunk], 
                                          skin[i:i+chunk], pts[i:i+chunk])
        pts_out.append(pts_chunk)
    pts = torch.cat(pts_out,0)
    return pts
def direct_qs_blend_skinning_chunk(Tmat, q_r, skin, pts):
    """
    bone: bs,B,10   - B gaussian ellipsoids
    rts: bs,B,3,4   - B ririd transforms, applied to bone coordinates (points attached to bones in world coords)
    pts: bs,N,3     - N 3d points
    skin: bs,N,B   - skinning matrix
    apply rts to bone coordinates, while computing blending globally
    """
    B = dq.shape[-2]
    N = pts.shape[-2]
    pts = pts.view(-1,N,3)

    # weighted dq and normalized it
    weighted_q_r = (skin[...,None] * q_r[:,None]).sum(2)   # (bs,N,B,1) * (bs,1,B,4)  = bs,N,B,4 -->bs,N,4
    q_r_normalized = dq_normalize(weighted_q_r)
    
    # translation
    Tmat_w = (skin[...,None] * Tmat[:,None]).sum(2) # bs,N,B,3-->bs,N,3
    pts = pts + Tmat_w

    # rotation
    xyz = transforms.quaternion_apply(q_r_normalized, pts)

    return xyz

def obj_to_cam(in_verts, Rmat, Tmat):
    """
    verts: ...,N,3
    Rmat:  ...,3,3
    Tmat:  ...,3 
    """
    verts = in_verts.clone()
    if verts.dim()==2: verts=verts[None]
    verts = verts.view(-1,verts.shape[1],3)
    Rmat = Rmat.view(-1,3,3).permute(0,2,1) # left multiply
    Tmat = Tmat.view(-1,1,3)
    
    verts =  verts.matmul(Rmat) + Tmat 
    verts = verts.reshape(in_verts.shape)
    return verts

def obj2cam_np(pts, Rmat, Tmat):
    """
    a wrapper for numpy array
    pts: ..., 3
    Rmat: 1,3,3
    Tmat: 1,3,3
    """
    pts_shape = pts.shape
    pts = torch.Tensor(pts).cuda().reshape(1,-1,3)
    pts = obj_to_cam(pts, Rmat,Tmat)
    return pts.view(pts_shape).cpu().numpy()

    
def K2mat(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = K[:,0]
    Kmat[:,1,1] = K[:,1]
    Kmat[:,0,2] = K[:,2]
    Kmat[:,1,2] = K[:,3]
    Kmat[:,2,2] = 1
    return Kmat

def mat2K(Kmat):
    """
    Kmat: ...,3,3
    """
    shape=Kmat.shape[:-2]
    Kmat = Kmat.view(-1,3,3)
    device = Kmat.device
    bs = Kmat.shape[0]

    K = torch.zeros(bs, 4, device=device)
    K[:,0] = Kmat[:,0,0]
    K[:,1] = Kmat[:,1,1]
    K[:,2] = Kmat[:,0,2]
    K[:,3] = Kmat[:,1,2]
    K = K.view(shape+(4,))
    return K

def Kmatinv(Kmat):
    """
    Kmat: ...,3,3
    """
    K = mat2K(Kmat)
    Kmatinv = K2inv(K)
    Kmatinv = Kmatinv.view(Kmat.shape)
    return Kmatinv

def K2inv(K):
    """
    K: ...,4
    """
    K = K.view(-1,4)
    device = K.device
    bs = K.shape[0]

    Kmat = torch.zeros(bs, 3, 3, device=device)
    Kmat[:,0,0] = 1./K[:,0]
    Kmat[:,1,1] = 1./K[:,1]
    Kmat[:,0,2] = -K[:,2]/K[:,0]
    Kmat[:,1,2] = -K[:,3]/K[:,1]
    Kmat[:,2,2] = 1
    return Kmat

def pinhole_cam(in_verts, K):
    """
    in_verts: ...,N,3
    K:        ...,4
    verts:    ...,N,3 in (x,y,Z)
    """
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[1],3)
    K = K.view(-1,4)

    Kmat = K2mat(K)
    Kmat = Kmat.permute(0,2,1)

    verts = verts.matmul(Kmat)
    verts_z = verts[:,:,2:3]
    verts_xy = verts[:,:,:2] / (1e-6+verts_z) # deal with neg z
    
    verts = torch.cat([verts_xy,verts_z],-1)
    verts = verts.reshape(in_verts.shape)
    return verts

def render_color(renderer, in_verts, faces, colors, texture_type='vertex'):
    """
    verts in ndc
    in_verts: ...,N,3/4
    faces: ...,N,3
    rendered: ...,4,...
    """
    import soft_renderer as sr
    verts = in_verts.clone()
    verts = verts.view(-1,verts.shape[-2],3)
    faces = faces.view(-1,faces.shape[-2],3)
    if texture_type=='vertex':  colors = colors.view(-1,colors.shape[-2],3)
    elif texture_type=='surface': colors = colors.view(-1,colors.shape[1],colors.shape[2],3)
    device=verts.device

    offset = torch.Tensor( renderer.transform.transformer._eye).to(device)[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]-offset
    verts_pre[:,:,1] = -1*verts_pre[:,:,1]  # pre-flip
    rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors,texture_type=texture_type))
    return rendered

def render_flow(renderer, verts, faces, verts_n):
    """
    rasterization
    verts in ndc
    verts: ...,N,3/4
    verts_n: ...,N,3/4
    faces: ...,N,3
    """
    verts = verts.view(-1,verts.shape[1],3)
    verts_n = verts_n.view(-1,verts_n.shape[1],3)
    faces = faces.view(-1,faces.shape[1],3)
    device=verts.device

    rendered_ndc_n = render_color(renderer, verts, faces, verts_n)
    _,_,h,w = rendered_ndc_n.shape
    rendered_sil = rendered_ndc_n[:,-1]

    ndc = np.meshgrid(range(w), range(h))
    ndc = torch.Tensor(ndc).to(device)[None]
    ndc[:,0] = ndc[:,0]*2 / (w-1) - 1
    ndc[:,1] = ndc[:,1]*2 / (h-1) - 1

    flow = rendered_ndc_n[:,:2] - ndc
    flow = flow.permute(0,2,3,1) # x,h,w,2
    flow = torch.cat([flow, rendered_sil[...,None]],-1)

    flow[rendered_sil<1]=0.
    flow[...,-1]=0. # discard the last channel
    return flow

def force_type(varlist):
    for i in range(len(varlist)):
        varlist[i] = varlist[i].type(varlist[0].dtype)
    return varlist

def tensor2array(tdict):
    adict={}
    for k,v in tdict.items():
        adict[k] = v.detach().cpu().numpy()
    return adict

def array2tensor(adict, device='cpu'):
    tdict={}
    for k,v in adict.items():
        try: 
            tdict[k] = torch.Tensor(v)
            if device != 'cpu': tdict[k] = tdict[k].to(device)
        except: pass # trimesh object
    return tdict

def raycast(xys, Rmat, Tmat, Kinv, near_far):
    """
    assuming xys and Rmat have same num of bs
    xys: bs, N, 3
    Rmat:bs, ...,3,3 
    Tmat:bs, ...,3, camera to root coord transform 
    Kinv:bs, ...,3,3 
    near_far:bs,2
    """
    Rmat, Tmat, Kinv, xys = force_type([Rmat, Tmat, Kinv, xys])
    Rmat = Rmat.view(-1,3,3)
    Tmat = Tmat.view(-1,1,3)
    Kinv = Kinv.view(-1,3,3)
    bs,nsample,_ = xys.shape
    device = Rmat.device

    xy1s = torch.cat([xys, torch.ones_like(xys[:,:,:1])],2)
    xyz3d = xy1s.matmul(Kinv.permute(0,2,1))
    ray_directions = xyz3d.matmul(Rmat)  # transpose -> right multiply
    ray_origins = -Tmat.matmul(Rmat) # transpose -> right multiply

    if near_far is not None:
        znear= (torch.ones(bs,nsample,1).to(device) * near_far[:,0,None,None]) 
        zfar = (torch.ones(bs,nsample,1).to(device) * near_far[:,1,None,None]) 
    else:
        lbound, ubound=[-1.5,1.5]

        znear= Tmat[:,:,-1:].repeat(1,nsample,1)+lbound
        zfar = Tmat[:,:,-1:].repeat(1,nsample,1)+ubound
        znear[znear<1e-5]=1e-5

    ray_origins = ray_origins.repeat(1,nsample,1)

    rmat_vec = Rmat.reshape(-1,1,9)
    tmat_vec = Tmat.reshape(-1,1,3)
    kinv_vec = Kinv.reshape(-1,1,9)
    rtk_vec = torch.cat([rmat_vec, tmat_vec, kinv_vec],-1) # x,21
    rtk_vec = rtk_vec.repeat(1,nsample,1)

    rays={'rays_o': ray_origins, 
          'rays_d': ray_directions,
          'near': znear,
          'far': zfar,
          'rtk_vec': rtk_vec,
          'xys': xys,
          'nsample': nsample,
          'bs': bs,
          }
    return rays

def sample_xy(img_size, bs, nsample, device, return_all=False, lineid=None):
    """
    rand_inds:  bs, ns
    xys:        bs, ns, 2
    """
    xygrid = np.meshgrid(range(img_size), range(img_size))  # w,h->hxw
    xygrid = torch.Tensor(xygrid).to(device)  # (x,y)
    xygrid = xygrid.permute(1,2,0).reshape(1,-1,2)  # 1,..., 2
    
    if return_all:
        xygrid = xygrid.repeat(bs,1,1)                  # bs,..., 2
        nsample = xygrid.shape[1]
        rand_inds=torch.Tensor(range(nsample))
        rand_inds=rand_inds[None].repeat(bs,1)
        xys = xygrid
    else:
        if lineid is None:
            probs = torch.ones(img_size**2).to(device) # 512*512 vs 128*64
            rand_inds = torch.multinomial(probs, bs*nsample, replacement=False)
            rand_inds = rand_inds.view(bs,nsample)
            xys = torch.stack([xygrid[0][rand_inds[i]] for i in range(bs)],0) # bs,ns,2
        else:
            probs = torch.ones(img_size).to(device) # 512*512 vs 128*64
            rand_inds = torch.multinomial(probs, bs*nsample, replacement=True)
            rand_inds = rand_inds.view(bs,nsample)
            xys = torch.stack([xygrid[0][rand_inds[i]] for i in range(bs)],0) # bs,ns,2
            xys[...,1] = xys[...,1] + lineid[:,None]
   
    rand_inds = rand_inds.long()
    # import pdb as p
    # p.set_trace()
    return rand_inds, xys

def chunk_rays(rays,start,delta):
    """
    rays: a dictionary
    """
    rays_chunk = {}
    for k,v in rays.items():
        if torch.is_tensor(v):
            v = v.view(-1, v.shape[-1])
            rays_chunk[k] = v[start:start+delta]
    return rays_chunk
        

def generate_bones(num_bones_x, num_bones, bound, device):
    """
    num_bones_x: bones along one direction
    bones: x**3,9
    """
    center =  torch.linspace(-bound, bound, num_bones_x).to(device)
    center =torch.meshgrid(center, center, center)
    center = torch.stack(center,0).permute(1,2,3,0).reshape(-1,3)
    center = center[:num_bones]
    
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = torch.zeros(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)
    return bones

def reinit_bones(model, mesh, num_bones, neudbs):
    """
    update the data of bones and nerf_body_rts[1].rgb without add new parameters
    num_bones: number of bones on the surface
    mesh: trimesh
    warning: ddp does not support adding/deleting parameters after construction
    """
    #TODO find another way to add/delete bones
    from kmeans_pytorch import kmeans
    device = model.device
    points = torch.Tensor(mesh.vertices).to(device)
    rthead = model.nerf_body_rts[1].rgb
    
    # reinit
    num_in = rthead[0].weight.shape[1]
    if neudbs:
        rthead = nn.Sequential(nn.Linear(num_in, 7*num_bones)).to(device) # 7
    else:
        rthead = nn.Sequential(nn.Linear(num_in, 6*num_bones)).to(device) # from 7 to 6
    torch.nn.init.xavier_uniform_(rthead[0].weight, gain=0.5)
    # torch.nn.init.normal_(rthead[0].weight, mean=0.0, std=0.001)
    torch.nn.init.zeros_(rthead[0].bias)

    if points.shape[0]<100:
        bound = model.latest_vars['obj_bound']
        bound = torch.Tensor(bound)[None]
        center = torch.rand(num_bones, 3) *  bound*2 - bound
    else:
        _, center = kmeans(X=points, num_clusters=num_bones, iter_limit=100,
                        tqdm_flag=False, distance='euclidean', device=device)
    center=center.to(device)
    orient =  torch.Tensor([[1,0,0,0]]).to(device)
    orient = orient.repeat(num_bones,1)
    scale = torch.zeros(num_bones,3).to(device)
    bones = torch.cat([center, orient, scale],-1)

    model.num_bones = num_bones
    num_output = model.nerf_body_rts[1].num_output
    bias_reinit =   rthead[0].bias.data
    weight_reinit=rthead[0].weight.data
    model.nerf_body_rts[1].rgb[0].bias.data[:num_bones*num_output] = bias_reinit
    model.nerf_body_rts[1].rgb[0].weight.data[:num_bones*num_output] = weight_reinit
    
    bones,_ = correct_bones(model, bones, inverse=True, neudbs=neudbs)
    model.bones.data[:num_bones] = bones
    model.nerf_models['bones'] = model.bones
    return


def lbs(bones, rts_fw, skin, xyz_in, backward=True):
    """
    bones: bs,B,10       - B gaussian ellipsoids indicating rest bone coordinates
    rts_fw: bs,B,12       - B rigid transforms, applied to the rest bones
    xyz_in: bs,N,3       - N 3d points after transforms in the root coordinates
    """
    B = bones.shape[-2]
    N = xyz_in.shape[-2]
    bs = rts_fw.shape[0]
    bones = bones.view(-1,B,10)
    xyz_in = xyz_in.view(-1,N,3)
    rts_fw = rts_fw.view(-1,B,12)# B,12
    rmat=rts_fw[:,:,:9]
    rmat=rmat.view(bs,B,3,3)
    tmat= rts_fw[:,:,9:12]
    rts_fw = torch.cat([rmat,tmat[...,None]],-1)
    rts_fw = rts_fw.view(-1,B,3,4)

    if backward:
        bones_dfm = bone_transform(bones, rts_fw, neudbs=False) # bone coordinates after deform
        rts_bw = rts_invert(rts_fw)
        xyz = blend_skinning(rts_bw, skin, xyz_in)
    else:
        xyz = blend_skinning(rts_fw, skin, xyz_in)
        bones_dfm = bone_transform(bones, rts_fw, neudbs=False) # bone coordinates after deform
    return xyz, bones_dfm

def correct_bones(model, bones_rst, inverse=False, neudbs=True): 
    # bones=>bones_rst
    bones_rst = bones_rst.clone()
    rest_pose_code =  model.rest_pose_code
    rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(model.device))
    rts_head = model.nerf_body_rts[1]
    bone_rts_rst = rts_head(rest_pose_code)[0] # 1,B*12-->1,B*8,get Jb* in paper
    if inverse:
        if neudbs:
            shape = bone_rts_rst.shape
            bone_rts_rst = bone_rts_rst.view(-1, model.opts.num_bones, 8)
            bone_rts_rst = dq_inverse(bone_rts_rst)
            bone_rts_rst = bone_rts_rst.view(shape)
        else: # use lbs
            bone_rts_rst = rtk_invert(bone_rts_rst, model.opts.num_bones)
    bones_rst = bone_transform(bones_rst, bone_rts_rst, neudbs, is_vec=True)[0] 
    return bones_rst, bone_rts_rst

def correct_rest_pose(opts, bone_rts_fw, bone_rts_rst, neudbs):
    """
    bone_rts_fw: bs, B, 8
    bone_rts_rst: B, 8
    get delta(Jb) = Jb(Jb*)^(-1)
    """
    # delta rts, deltaJb in eq 9
    bone_rts_fw = bone_rts_fw.clone()
    rts_shape = bone_rts_fw.shape
    if neudbs:
        bone_rts_rst = bone_rts_rst.view(-1, opts.num_bones, 8)
        bone_rts_fw = bone_rts_fw.view(-1, opts.num_bones, 8)
        bone_rts_rst_inv = dq_inverse(bone_rts_rst)

        bone_rts_rst_inv = bone_rts_rst_inv.repeat(rts_shape[0],rts_shape[1],1)
        bone_rts_fw = dq_mul(bone_rts_rst_inv, bone_rts_fw)    #rtk_compose(bone_rts_rst_inv, bone_rts_fw)
        return bone_rts_fw.view(rts_shape)
    else:
        bone_rts_rst_inv = rtk_invert(bone_rts_rst, opts.num_bones)
        bone_rts_rst_inv = bone_rts_rst_inv.repeat(rts_shape[0],rts_shape[1],1)
        bone_rts_fw =     rtk_compose(bone_rts_rst_inv, bone_rts_fw)
        return bone_rts_fw

def warp_bw(opts, model, rt_dict, query_xyz_chunk, embedid):
    """
    only used in mesh extraction
    embedid: embedding id
    """
    chunk = query_xyz_chunk.shape[0]
    query_time = torch.ones(chunk,1).to(model.device)*embedid
    query_time = query_time.long()
    if opts.flowbw:
        # flowbw
        xyz_embedded = model.embedding_xyz(query_xyz_chunk)
        time_embedded = model.pose_code(query_time)#[:,0]
        xyztime_embedded = torch.cat([xyz_embedded, time_embedded],1)

        flowbw_chunk = model.nerf_flowbw(xyztime_embedded, xyz=query_xyz_chunk)
        query_xyz_chunk += flowbw_chunk
    elif (opts.lbs or opts.neudbs):
        # backward skinning: from camera space to canonical
        bones_rst = model.bones
        bone_rts_fw = model.nerf_body_rts(query_time)   #Jb = MLPJ (pose code)
        # update bones
        bones_rst, bone_rts_rst = correct_bones(model, bones_rst, neudbs=opts.neudbs)
        bone_rts_fw = correct_rest_pose(opts, bone_rts_fw, bone_rts_rst, opts.neudbs)

        query_xyz_chunk = query_xyz_chunk[:,None]

        if opts.nerf_skin:
            nerf_skin = model.nerf_skin
        else:
            nerf_skin = None
        time_embedded = model.pose_code(query_time)
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, opts.neudbs, is_vec=True)

        skin_backward = gauss_mlp_skinning(query_xyz_chunk, model.embedding_xyz,
                   bones_dfm, time_embedded, nerf_skin, skin_aux=model.skin_aux )   #get W for lbs

        if opts.nerf_dis:
            nerf_dis = model.nerf_dis
        else:
            nerf_dis = None
        if opts.lbs:
            query_xyz_chunk,bones_dfm = lbs(bones_rst, 
                                        bone_rts_fw,
                                        skin_backward,
                                        query_xyz_chunk)  # add displacement field
        elif opts.neudbs:
            query_xyz_chunk,bones_dfm,_ = neu_dbs(bones_rst, 
                                        bone_rts_fw,
                                        skin_backward,
                                        query_xyz_chunk, nerf_dis, model.embedding_xyz, time_embedded)  # add displacement field

        query_xyz_chunk = query_xyz_chunk[:,0]
        rt_dict['bones'] = bones_dfm 
    return query_xyz_chunk, rt_dict
        
def warp_fw(opts, model, rt_dict, vertices, embedid):
    """
    only used in mesh extraction
    """
    num_pts = vertices.shape[0]
    query_time = torch.ones(num_pts,1).long().to(model.device)*embedid
    pts_can=torch.Tensor(vertices).to(model.device)
    if opts.flowbw:
        # forward flow
        pts_can_embedded = model.embedding_xyz(pts_can)
        time_embedded = model.pose_code(query_time)#[:,0]
        ptstime_embedded = torch.cat([pts_can_embedded, time_embedded],1)

        pts_dfm = pts_can + model.nerf_flowfw(ptstime_embedded, xyz=pts_can)
    elif (opts.lbs or opts.neudbs):
        # forward skinning" canonical to camera
        pts_can = pts_can[:,None]
        bones_rst = model.bones
        bone_rts_fw = model.nerf_body_rts(query_time)
        bones_rst, bone_rts_rst = correct_bones(model, bones_rst, neudbs=opts.neudbs)
        bone_rts_fw = correct_rest_pose(opts, bone_rts_fw, bone_rts_rst, opts.neudbs)
        
        if opts.nerf_skin:
            nerf_skin = model.nerf_skin
        else:
            nerf_skin = None
        rest_pose_code =  model.rest_pose_code
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        # query_time = torch.ones(chunk,1).to(model.device)*embedid
        skin_forward = gauss_mlp_skinning(pts_can, model.embedding_xyz, bones_rst, 
                            rest_pose_code, nerf_skin, skin_aux=model.skin_aux)
        if opts.nerf_dis:
            nerf_dis = model.nerf_dis
        else:
            nerf_dis = None
        if opts.lbs:
            pts_dfm,bones_dfm = lbs(bones_rst, bone_rts_fw, skin_forward, 
                    pts_can, backward=False)
        elif opts.neudbs:
            pts_dfm,bones_dfm,_ = neu_dbs(bones_rst, bone_rts_fw, skin_forward, 
                    pts_can, nerf_dis, model.embedding_xyz, rest_pose_code, backward=False)
        pts_dfm = pts_dfm[:,0]
        rt_dict['bones'] = bones_dfm
    vertices = pts_dfm.cpu().numpy()
    return vertices, rt_dict
    
def canonical2ndc(model, dp_canonical_pts, rtk, kaug, embedid):
    """
    dp_canonical_pts: 5004,3, pts in the canonical space of each video
    dp_px: bs, 5004, 3
    """
    Rmat = rtk[:,:3,:3]
    Tmat = rtk[:,:3,3]
    Kmat = K2mat(rtk[:,3,:])
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    Kinv = Kmatinv(Kaug.matmul(Kmat))
    K = mat2K(Kmatinv(Kinv))
    bs = Kinv.shape[0]
    npts = dp_canonical_pts.shape[0]

    # projection
    dp_canonical_pts = dp_canonical_pts[None]
    if model.opts.flowbw:
        time_embedded = model.pose_code(embedid)
        time_embedded = time_embedded.repeat(1,npts, 1)
        dp_canonical_embedded = model.embedding_xyz(dp_canonical_pts)[None]
        dp_canonical_embedded = dp_canonical_embedded.repeat(bs,1,1)
        dp_canonical_embedded = torch.cat([dp_canonical_embedded, time_embedded], -1)
        dp_deformed_flo = model.nerf_flowfw(dp_canonical_embedded, xyz=dp_canonical_pts)
        dp_deformed_pts = dp_canonical_pts + dp_deformed_flo
    else:
        dp_deformed_pts = dp_canonical_pts.repeat(bs,1,1)
    dp_cam_pts = obj_to_cam(dp_deformed_pts, Rmat, Tmat) 
    dp_px = pinhole_cam(dp_cam_pts,K)
    return dp_px 

def get_near_far(near_far, vars_np, tol_fac=1.2, pts=None):
    """
    pts:        point coordinate N,3
    near_far:   near and far plane M,2
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    tol_fac     tolerance factor
    """
    if pts is None:
        #pts = vars_np['mesh_rest'].vertices
        # turn points to bounding box
        #print('bug here!!!')
        #print(vars_np['mesh_rest'].bounds)
        pts = trimesh.bounds.corners(vars_np['mesh_rest'].bounds)

    device = near_far.device
    rtk = torch.Tensor(vars_np['rtk']).to(device)
    idk = torch.Tensor(vars_np['idk']).to(device)

    pts = pts_to_view(pts, rtk, device)

    pmax = pts[...,-1].max(-1)[0]
    pmin = pts[...,-1].min(-1)[0]
    delta = (pmax - pmin)*(tol_fac-1)

    near= pmin-delta
    far = pmax+delta

    near_far[idk==1,0] = torch.clamp(near[idk==1], min=1e-3)
    near_far[idk==1,1] = torch.clamp( far[idk==1], min=1e-3)
    return near_far

def pts_to_view(pts, rtk, device):
    """
    object to camera coordinates
    pts:        point coordinate N,3
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    """
    M = rtk.shape[0]
    out_pts = []
    chunk=100
    for i in range(0,M,chunk):
        rtk_sub = rtk[i:i+chunk]
        pts_sub = torch.Tensor(np.tile(pts[None],
                        (len(rtk_sub),1,1))).to(device) # M,N,3
        pts_sub = obj_to_cam(pts_sub,  rtk_sub[:,:3,:3], 
                                       rtk_sub[:,:3,3])
        pts_sub = pinhole_cam(pts_sub, rtk_sub[:,3])
        out_pts.append(pts_sub)
    out_pts = torch.cat(out_pts, 0)
    return out_pts

def compute_point_visibility(pts, vars_np, device):
    """
    pts:        point coordinate N,3
    rtk:        object to camera transform, M,4,4
    idk:        indicator of obsered or not M
    **deprecated** due to K vars_tensor['rtk'] may not be consistent
    """
    vars_tensor = array2tensor(vars_np, device=device)
    rtk = vars_tensor['rtk']
    idk = vars_tensor['idk']
    vis = vars_tensor['vis']
    
    pts = pts_to_view(pts, rtk, device) # T, N, 3
    h,w = vis.shape[1:]

    vis = vis[:,None]
    xy = pts[:,None,:,:2] 
    xy[...,0] = xy[...,0]/w*2 - 1
    xy[...,1] = xy[...,1]/h*2 - 1

    # grab the visibility value in the mask and sum over frames
    vis = F.grid_sample(vis, xy)[:,0,0]
    vis = (idk[:,None]*vis).sum(0)
    vis = (vis>0).float() # at least seen in one view
    return vis


def near_far_to_bound(near_far):
    """
    near_far: T, 2 on cuda
    bound: float
    this can only be used for a single video (and for approximation)
    """
    bound=(near_far[:,1]-near_far[:,0]).mean() / 2
    bound = bound.detach().cpu().numpy()
    return bound


def rot_angle(mat):
    """
    rotation angle of rotation matrix 
    rmat: ..., 3,3
    """
    eps=1e-4
    cos = (  mat[...,0,0] + mat[...,1,1] + mat[...,2,2] - 1 )/2
    cos = cos.clamp(-1+eps,1-eps)
    angle = torch.acos(cos)
    return angle

def match2coords(match, w_rszd):
    tar_coord = torch.cat([match[:,None]%w_rszd, match[:,None]//w_rszd],-1)
    tar_coord = tar_coord.float()
    return tar_coord
    
def match2flo(match, w_rszd, img_size, warp_r, warp_t, device):
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.matmul(warp_r[:2,:2]) + warp_r[None,:2,2]
    tar_coord = match2coords(match, w_rszd)
    tar_coord = tar_coord.matmul(warp_t[:2,:2]) + warp_t[None,:2,2]

    flo_dp = (tar_coord - ref_coord) / img_size * 2 # [-2,2]
    flo_dp = flo_dp.view(w_rszd, w_rszd, 2)
    flo_dp = flo_dp.permute(2,0,1)

    xygrid = sample_xy(w_rszd, 1, 0, device, return_all=True)[1] # scale to img_size
    xygrid = xygrid * float(img_size/w_rszd)
    warp_r_inv = Kmatinv(warp_r)
    xygrid = xygrid.matmul(warp_r_inv[:2,:2]) + warp_r_inv[None,:2,2]
    xygrid = xygrid / w_rszd * 2 - 1 
    flo_dp = F.grid_sample(flo_dp[None], xygrid.view(1,w_rszd,w_rszd,2))[0]
    return flo_dp

def compute_flow_cse(cse_a,cse_b, warp_a, warp_b, img_size):
    """
    compute the flow between two frames under cse feature matching
    assuming two feature images have the same dimension (also rectangular)
    cse:        16,h,w, feature image
    flo_dp:     2,h,w
    """
    _,_,w_rszd = cse_a.shape
    hw_rszd = w_rszd*w_rszd
    device = cse_a.device

    cost = (cse_b[:,None,None] * cse_a[...,None,None]).sum(0)
    _,match_a = cost.view(hw_rszd, hw_rszd).max(1)
    _,match_b = cost.view(hw_rszd, hw_rszd).max(0)

    flo_a = match2flo(match_a, w_rszd, img_size, warp_a, warp_b, device)
    flo_b = match2flo(match_b, w_rszd, img_size, warp_b, warp_a, device)
    return flo_a, flo_b

def compute_flow_geodist(dp_refr,dp_targ, geodists):
    """
    compute the flow between two frames under geodesic distance matching
    dps:        h,w, canonical surface mapping index
    geodists    N,N, distance matrix
    flo_dp:     2,h,w
    """
    h_rszd,w_rszd = dp_refr.shape
    hw_rszd = h_rszd*w_rszd
    device = dp_refr.device
    chunk = 1024

    # match: hw**2
    match = torch.zeros(hw_rszd).to(device)
    for i in range(0,hw_rszd,chunk):
        chunk_size = len(dp_refr.view(-1,1)[i:i+chunk] )
        dp_refr_sub = dp_refr.view(-1,1)[i:i+chunk].repeat(1,hw_rszd).view(-1,1)
        dp_targ_sub = dp_targ.view(1,-1)        .repeat(chunk_size,1).view(-1,1)
        match_sub = geodists[dp_refr_sub, dp_targ_sub]
        dis_geo_sub,match_sub = match_sub.view(-1, hw_rszd).min(1)
        #match_sub[dis_geo_sub>0.1] = 0
        match[i:i+chunk] = match_sub

    # cx,cy
    tar_coord = match2coords(match, w_rszd)
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.view(h_rszd, w_rszd, 2)
    tar_coord = tar_coord.view(h_rszd, w_rszd, 2)
    flo_dp = (tar_coord - ref_coord) / w_rszd * 2 # [-2,2]
    match = match.view(h_rszd, w_rszd)
    flo_dp[match==0] = 0
    flo_dp = flo_dp.permute(2,0,1)
    return flo_dp

def compute_flow_geodist_old(dp_refr,dp_targ, geodists):
    """
    compute the flow between two frames under geodesic distance matching
    dps:        h,w, canonical surface mapping index
    geodists    N,N, distance matrix
    flo_dp:     2,h,w
    """
    h_rszd,w_rszd = dp_refr.shape
    hw_rszd = h_rszd*w_rszd
    device = dp_refr.device
    dp_refr = dp_refr.view(-1,1).repeat(1,hw_rszd).view(-1,1)
    dp_targ = dp_targ.view(1,-1).repeat(hw_rszd,1).view(-1,1)

    match = geodists[dp_refr, dp_targ]
    dis_geo,match = match.view(hw_rszd, hw_rszd).min(1)
    #match[dis_geo>0.1] = 0

    # cx,cy
    tar_coord = match2coords(match, w_rszd)
    ref_coord = sample_xy(w_rszd, 1, 0, device, return_all=True)[1].view(-1,2)
    ref_coord = ref_coord.view(h_rszd, w_rszd, 2)
    tar_coord = tar_coord.view(h_rszd, w_rszd, 2)
    flo_dp = (tar_coord - ref_coord) / w_rszd * 2 # [-2,2]
    match = match.view(h_rszd, w_rszd)
    flo_dp[match==0] = 0
    flo_dp = flo_dp.permute(2,0,1)
    return flo_dp



def fb_flow_check(flo_refr, flo_targ, img_refr, img_targ, dp_thrd, 
                    save_path=None):
    """
    apply forward backward consistency check on flow fields
    flo_refr: 2,h,w forward flow
    flo_targ: 2,h,w backward flow
    fberr:    h,w forward backward error
    """
    h_rszd, w_rszd = flo_refr.shape[1:]
    # clean up flow
    flo_refr = flo_refr.permute(1,2,0).cpu().numpy()
    flo_targ = flo_targ.permute(1,2,0).cpu().numpy()
    flo_refr_mask = np.linalg.norm(flo_refr,2,-1)>0 # this also removes 0 flows
    flo_targ_mask = np.linalg.norm(flo_targ,2,-1)>0
    flo_refr_px = flo_refr * w_rszd / 2
    flo_targ_px = flo_targ * w_rszd / 2

    #fb check
    x0,y0  =np.meshgrid(range(w_rszd),range(h_rszd))
    hp0 = np.stack([x0,y0],-1) # screen coord

    flo_fb = warp_flow(hp0 + flo_targ_px, flo_refr_px) - hp0
    flo_fb = 2*flo_fb/w_rszd
    fberr_fw = np.linalg.norm(flo_fb, 2,-1)
    fberr_fw[~flo_refr_mask] = 0

    flo_bf = warp_flow(hp0 + flo_refr_px, flo_targ_px) - hp0
    flo_bf = 2*flo_bf/w_rszd
    fberr_bw = np.linalg.norm(flo_bf, 2,-1)
    fberr_bw[~flo_targ_mask] = 0

    if save_path is not None:
        # vis
        thrd_vis = 0.01
        img_refr = F.interpolate(img_refr, (h_rszd, w_rszd), mode='bilinear')[0]
        img_refr = img_refr.permute(1,2,0).cpu().numpy()[:,:,::-1]
        img_targ = F.interpolate(img_targ, (h_rszd, w_rszd), mode='bilinear')[0]
        img_targ = img_targ.permute(1,2,0).cpu().numpy()[:,:,::-1]
        flo_refr[:,:,0] = (flo_refr[:,:,0] + 2)/2
        flo_targ[:,:,0] = (flo_targ[:,:,0] - 2)/2
        flo_refr[fberr_fw>thrd_vis]=0.
        flo_targ[fberr_bw>thrd_vis]=0.
        flo_refr[~flo_refr_mask]=0.
        flo_targ[~flo_targ_mask]=0.
        img = np.concatenate([img_refr, img_targ], 1)
        flo = np.concatenate([flo_refr, flo_targ], 1)
        imgflo = cat_imgflo(img, flo)
        imgcnf = np.concatenate([fberr_fw, fberr_bw],1)
        imgcnf = np.clip(imgcnf, 0, dp_thrd)*(255/dp_thrd)
        imgcnf = np.repeat(imgcnf[...,None],3,-1)
        imgcnf = cv2.resize(imgcnf, imgflo.shape[::-1][1:])
        imgflo_cnf = np.concatenate([imgflo, imgcnf],0)
        cv2.imwrite(save_path, imgflo_cnf)
    return fberr_fw, fberr_bw


def mask_aug(rendered):
    lb = 0.1;    ub = 0.3
    _,h,w=rendered.shape
    if np.random.binomial(1,0.5):
        sx = int(np.random.uniform(lb*w,ub*w))
        sy = int(np.random.uniform(lb*h,ub*h))
        cx = int(np.random.uniform(sx,w-sx))
        cy = int(np.random.uniform(sy,h-sy))
        feat_mean = rendered.mean(-1).mean(-1)[:,None,None]
        rendered[:,cx-sx:cx+sx,cy-sy:cy+sy] = feat_mean
    return rendered

def process_so3_seq(rtk_seq, vis=False, smooth=True):
    """
    rtk_seq, bs, N, 13 including
    {scoresx1, rotationsx9, translationsx3}
    """
    from utils.io import draw_cams
    scores =rtk_seq[...,0]
    bs,N = scores.shape
    rmat =  rtk_seq[...,1:10]
    tmat = rtk_seq[:,0,10:13]
    rtk_raw = rtk_seq[:,0,13:29].reshape((-1,4,4))
   
    distribution = torch.Tensor(scores).softmax(1)
    entropy = (-distribution.log() * distribution).sum(1)

    if vis:
        # draw distribution
        obj_scale = 3
        cam_space = obj_scale * 0.2
        tmat_raw = np.tile(rtk_raw[:,None,:3,3], (1,N,1))
        scale_factor = obj_scale/tmat_raw[...,-1].mean()
        tmat_raw *= scale_factor
        tmat_raw = tmat_raw.reshape((bs,12,-1,3))
        tmat_raw[...,-1] += np.linspace(-cam_space, cam_space,12)[None,:,None]
        tmat_raw = tmat_raw.reshape((bs,-1,3))
        # bs, tiltxae
        all_rts = np.concatenate([rmat, tmat_raw],-1)
        all_rts = np.transpose(all_rts.reshape(bs,N,4,3), [0,1,3,2])
    
        for i in range(bs):
            top_idx = scores[i].argsort()[-30:]
            top_rt = all_rts[i][top_idx]
            top_score = scores[i][top_idx]
            top_score = (top_score - top_score.min())/(top_score.max()-top_score.min())
            mesh = draw_cams(top_rt, color_list = top_score)
            mesh.export('tmp/%d.obj'%(i))
   
    if smooth:
        # graph cut scores, bsxN
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
        graph = dcrf.DenseCRF2D(bs, 1, N)  # width, height, nlabels
        unary = unary_from_softmax(distribution.numpy().T.copy())
        graph.setUnaryEnergy(unary)
        grid = rmat[0].reshape((N,3,3))
        drot = np.matmul(grid[None], np.transpose(grid[:,None], (0,1,3,2)))
        drot = rot_angle(torch.Tensor(drot))
        compat = (-2*(drot).pow(2)).exp()*10
        compat = compat.numpy()
        graph.addPairwiseGaussian(sxy=10, compat=compat)

        Q = graph.inference(100)
        scores = np.asarray(Q).T

    # argmax
    idx_max = scores.argmax(-1)
    rmat = rmat[0][idx_max]

    rmat = rmat.reshape((-1,9))
    rts = np.concatenate([rmat, tmat],-1)
    rts = rts.reshape((bs,1,-1))

    # post-process se3
    root_rmat = rts[:,0,:9].reshape((-1,3,3))
    root_tmat = rts[:,0,9:12]
    
    rmat = rtk_raw[:,:3,:3]
    tmat = rtk_raw[:,:3,3]
    tmat = tmat + np.matmul(rmat, root_tmat[...,None])[...,0]
    rmat = np.matmul(rmat, root_rmat)
    rtk_raw[:,:3,:3] = rmat
    rtk_raw[:,:3,3] = tmat
   
    if vis:
        # draw again
        pdb.set_trace()
        rtk_vis = rtk_raw.copy()
        rtk_vis[:,:3,3] *= scale_factor
        mesh = draw_cams(rtk_vis)
        mesh.export('tmp/final.obj')
    return rtk_raw

def align_sim3(rootlist_a, rootlist_b, is_inlier=None, err_valid=None):
    """
    nx4x4 matrices
    is_inlier: n
    """
#    ta = np.matmul(-np.transpose(rootlist_a[:,:3,:3],[0,2,1]), 
#                                 rootlist_a[:,:3,3:4])
#    ta = ta[...,0].T
#    tb = np.matmul(-np.transpose(rootlist_b[:,:3,:3],[0,2,1]), 
#                                 rootlist_b[:,:3,3:4])
#    tb = tb[...,0].T
#    dso3,dtrn,dscale=umeyama_alignment(tb, ta,with_scale=False)
#    
#    dscale = np.linalg.norm(rootlist_a[0,:3,3],2,-1) /\
#             np.linalg.norm(rootlist_b[0,:3,3],2,-1)
#    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3.T[None])
#    rootlist_b[:,:3,3:4] = rootlist_b[:,:3,3:4] - \
#            np.matmul(rootlist_b[:,:3,:3], dtrn[None,:,None]) 

    dso3 = np.matmul(np.transpose(rootlist_b[:,:3,:3],(0,2,1)),
                        rootlist_a[:,:3,:3])
    dscale = np.linalg.norm(rootlist_a[:,:3,3],2,-1)/\
            np.linalg.norm(rootlist_b[:,:3,3],2,-1)

    # select inliers to fit 
    if is_inlier is not None:
        if is_inlier.sum() == 0:
            is_inlier[np.argmin(err_valid)] = True
        dso3 = dso3[is_inlier]
        dscale = dscale[is_inlier]

    dso3 = R.from_matrix(dso3).mean().as_matrix()
    rootlist_b[:,:3,:3] = np.matmul(rootlist_b[:,:3,:3], dso3[None])

    dscale = dscale.mean()
    rootlist_b[:,:3,3] = rootlist_b[:,:3,3] * dscale

    so3_err = np.matmul(rootlist_a[:,:3,:3], 
            np.transpose(rootlist_b[:,:3,:3],[0,2,1]))
    so3_err = rot_angle(torch.Tensor(so3_err))
    so3_err = so3_err / np.pi*180
    so3_err_max = so3_err.max()
    so3_err_mean = so3_err.mean()
    so3_err_med = np.median(so3_err)
    so3_err_std = np.asarray(so3_err.std())
    print(so3_err)
    print('max  so3 error (deg): %.1f'%(so3_err_max))
    print('med  so3 error (deg): %.1f'%(so3_err_med))
    print('mean so3 error (deg): %.1f'%(so3_err_mean))
    print('std  so3 error (deg): %.1f'%(so3_err_std))

    return rootlist_b

def align_sfm_sim3(aux_seq, datasets):
    from utils.io import draw_cams, load_root
    for dataset in datasets:
        seqname = dataset.imglist[0].split('/')[-2]

        # only process dataset with rtk_path input
        if dataset.has_prior_cam:
            root_dir = dataset.rtklist[0][:-9]
            root_sfm = load_root(root_dir, 0)[:-1] # excluding the last

            # split predicted root into multiple sequences
            seq_idx = [seqname == i.split('/')[-2] for i in aux_seq['impath']]
            root_pred = aux_seq['rtk'][seq_idx]
            is_inlier = aux_seq['is_valid'][seq_idx]
            err_valid = aux_seq['err_valid'][seq_idx]
            # only use certain ones to match
            #pdb.set_trace()
            #mesh = draw_cams(root_sfm, color='gray')
            #mesh.export('0.obj')
            
            # pre-align the center according to cat mask
            root_sfm = visual_hull_align(root_sfm, 
                    aux_seq['kaug'][seq_idx],
                    aux_seq['masks'][seq_idx])

            root_sfm = align_sim3(root_pred, root_sfm, 
                    is_inlier=is_inlier, err_valid=err_valid)
            # only modify rotation
            #root_pred[:,:3,:3] = root_sfm[:,:3,:3]
            root_pred = root_sfm
            
            aux_seq['rtk'][seq_idx] = root_pred
            aux_seq['is_valid'][seq_idx] = True
        else:
            print('not aligning %s, no rtk path in config file'%seqname)

def visual_hull_align(rtk, kaug, masks):
    """
    input: array
    output: array
    """
    rtk = torch.Tensor(rtk)
    kaug = torch.Tensor(kaug)
    masks = torch.Tensor(masks)
    num_view,h,w = masks.shape
    grid_size = 64
   
    if rtk.shape[0]!=num_view:
        print('rtk size mismtach: %d vs %d'%(rtk.shape[0], num_view))
        rtk = rtk[:num_view]
        
    rmat = rtk[:,:3,:3]
    tmat = rtk[:,:3,3:]

    Kmat = K2mat(rtk[:,3])
    Kaug = K2inv(kaug) # p = Kaug Kmat P
    kmat = mat2K(Kaug.matmul(Kmat))

    rmatc = rmat.permute((0,2,1))
    tmatc = -rmatc.matmul(tmat)

    bound = tmatc.norm(2,-1).mean()
    pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
    query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
    query_yxz = torch.Tensor(query_yxz).view(-1, 3)
    query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

    score_xyz = []
    chunk = 1000
    for i in range(0,len(query_xyz),chunk):
        query_xyz_chunk = query_xyz[None, i:i+chunk].repeat(num_view, 1,1)
        query_xyz_chunk = obj_to_cam(query_xyz_chunk, rmat, tmat)
        query_xyz_chunk = pinhole_cam(query_xyz_chunk, kmat)

        query_xy = query_xyz_chunk[...,:2]
        query_xy[...,0] = query_xy[...,0]/w*2-1
        query_xy[...,1] = query_xy[...,1]/h*2-1

        # sum over time
        score = F.grid_sample(masks[:,None], query_xy[:,None])[:,0,0]
        score = score.sum(0)
        score_xyz.append(score)

    # align the center
    score_xyz = torch.cat(score_xyz)
    center = query_xyz[score_xyz>0.8*num_view]
    print('%d points used to align center'% (len(center)) )
    center = center.mean(0)
    tmatc = tmatc - center[None,:,None]
    tmat = np.matmul(-rmat, tmatc)
    rtk[:,:3,3:] = tmat

    return rtk

def ood_check_cse(dp_feats, dp_embed, dp_idx):
    """
    dp_feats: bs,16,h,w
    dp_idx:   bs, h,w
    dp_embed: N,16
    valid_list bs
    """
    bs,_,h,w = dp_feats.shape
    N,_ = dp_embed.shape
    device = dp_feats.device
    dp_idx = F.interpolate(dp_idx.float()[None], (h,w), mode='nearest').long()[0]
    
    ## dot product 
    #pdb.set_trace()
    #err_list = []
    #err_threshold = 0.05
    #for i in range(bs):
    #    err = 1- (dp_embed[dp_idx[i]]*dp_feats[i].permute(1,2,0)).sum(-1)
    #    err_list.append(err)

    # fb check
    err_list = []
    err_threshold = 12
    # TODO no fb check
    #err_threshold = 100
    for i in range(bs):
        # use chunk
        chunk = 5000
        max_idx = torch.zeros(N).to(device)
        for j in range(0,N,chunk):
            costmap = (dp_embed.view(N,16,1)[j:j+chunk]*\
                    dp_feats[i].view(1,16,h*w)).sum(-2)
            max_idx[j:j+chunk] = costmap.argmax(-1)  #  N
    
        rpj_idx = max_idx[dp_idx[i]]
        rpj_coord = torch.stack([rpj_idx % w, rpj_idx//w],-1)
        ref_coord = sample_xy(w, 1, 0, device, return_all=True)[1].view(h,w,2)
        err = (rpj_coord - ref_coord).norm(2,-1) 
        err_list.append(err)

    valid_list = []
    error_list = []
    for i in range(bs):
        err = err_list[i]
        mean_error = err[dp_idx[i]!=0].mean()
        is_valid = mean_error < err_threshold
        error_list.append( mean_error)
        valid_list.append( is_valid  )
        #cv2.imwrite('tmp/%05d.png'%i, (err/mean_error).cpu().numpy()*100)
        #print(i); print(mean_error)
    error_list = torch.stack(error_list,0)
    valid_list = torch.stack(valid_list,0)

    return valid_list, error_list

def bbox_dp2rnd(bbox, kaug):
    """
    bbox: bs, 4
    kaug: bs, 4
    cropab2: bs, 3,3, transformation from dp bbox to rendered bbox coords
    """
    cropa2im = torch.cat([(bbox[:,2:] - bbox[:,:2]) / 112., 
                           bbox[:,:2]],-1)
    cropa2im = K2mat(cropa2im)
    im2cropb = K2inv(kaug) 
    cropa2b = im2cropb.matmul(cropa2im)
    return cropa2b
            



def resample_dp(dp_feats, dp_bbox, kaug, target_size):
    """
    dp_feats: bs, 16, h,w
    dp_bbox:  bs, 4
    kaug:     bs, 4
    """
    # if dp_bbox are all zeros, just do the resizing
    if dp_bbox.abs().sum()==0:
        dp_feats_rsmp = F.interpolate(dp_feats, (target_size, target_size),
                                                            mode='bilinear')
    else:
        dp_size = dp_feats.shape[-1]
        device = dp_feats.device

        dp2rnd = bbox_dp2rnd(dp_bbox, kaug)
        rnd2dp = Kmatinv(dp2rnd)
        xygrid = sample_xy(target_size, 1, 0, device, return_all=True)[1] 
        xygrid = xygrid.matmul(rnd2dp[:,:2,:2]) + rnd2dp[:,None,:2,2]
        xygrid = xygrid / dp_size * 2 - 1 
        dp_feats_rsmp = F.grid_sample(dp_feats, xygrid.view(-1,target_size,target_size,2))
    return dp_feats_rsmp


def vrender_flo(weights_coarse, xyz_coarse_target, xys, img_size):
    """
    weights_coarse:     ..., ndepth
    xyz_coarse_target:  ..., ndepth, 3
    flo_coarse:         ..., 2
    flo_valid:          ..., 1
    """
    # render flow 
    weights_coarse = weights_coarse.clone()
    xyz_coarse_target = xyz_coarse_target.clone()

    # bs, nsamp, -1, x
    weights_shape = weights_coarse.shape
    xyz_coarse_target = xyz_coarse_target.view(weights_shape+(3,))
    xy_coarse_target = xyz_coarse_target[...,:2]

    # deal with negative z
    invalid_ind = torch.logical_or(xyz_coarse_target[...,-1]<1e-5,
                           xy_coarse_target.norm(2,-1).abs()>2*img_size)
    weights_coarse[invalid_ind] = 0.
    xy_coarse_target[invalid_ind] = 0.

    # renormalize
    weights_coarse = weights_coarse/(1e-9+weights_coarse.sum(-1)[...,None])

    # candidate motion vector
    xys_unsq = xys.view(weights_shape[:-1]+(1,2))
    flo_coarse = xy_coarse_target - xys_unsq
    flo_coarse =  weights_coarse[...,None] * flo_coarse
    flo_coarse = flo_coarse.sum(-2)

    ## candidate target point
    #xys_unsq = xys.view(weights_shape[:-1]+(2,))
    #xy_coarse_target = weights_coarse[...,None] * xy_coarse_target
    #xy_coarse_target = xy_coarse_target.sum(-2)
    #flo_coarse = xy_coarse_target - xys_unsq

    flo_coarse = flo_coarse/img_size * 2
    flo_valid = (invalid_ind.sum(-1)==0).float()[...,None]
    return flo_coarse, flo_valid

def diff_flo(pts_target, xys, img_size):
    """
    pts_target:         ..., 1, 2
    xys:                ..., 2
    flo_coarse:         ..., 2
    flo_valid:          ..., 1
    """

    # candidate motion vector
    pts_target = pts_target.view(xys.shape)
    flo_coarse = pts_target - xys
    flo_coarse = flo_coarse/img_size * 2
    return flo_coarse

def fid_reindex(fid, num_vids, vid_offset):
    """
    re-index absolute frameid {0,....N} to subsets of video id and relative frameid
    fid: N absolution id
    vid: N video id
    tid: N relative id
    """
    tid = torch.zeros_like(fid).float()
    vid = torch.zeros_like(fid)
    max_ts = (vid_offset[1:] - vid_offset[:-1]).max()
    for i in range(num_vids):
        assign = torch.logical_and(fid>=vid_offset[i],
                                    fid<vid_offset[i+1])
        vid[assign] = i
        tid[assign] = fid[assign].float() - vid_offset[i]
        doffset = vid_offset[i+1] - vid_offset[i]
        tid[assign] = (tid[assign] - doffset/2)/max_ts*2
        #tid[assign] = 2*(tid[assign] / doffset)-1
        #tid[assign] = (tid[assign] - doffset/2)/1000.
    return vid, tid
