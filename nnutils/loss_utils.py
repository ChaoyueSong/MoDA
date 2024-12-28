# Modified from https://github.com/facebookresearch/banmo

import pdb
import trimesh
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from nnutils.geom_utils import rot_angle, mat2K, Kmatinv, obj_to_cam, \
                                pinhole_cam, lbs, neu_dbs, gauss_mlp_skinning, evaluate_mlp


def nerf_gradient(mlp, embed, pts, use_xyz=False,code=None, sigma_only=False):
    """
    gradient of mlp params wrt pts
    """
    pts.requires_grad_(True)
    pts_embedded = embed(pts)
    if use_xyz: xyz=pts
    else: xyz=None
    y = evaluate_mlp(mlp, pts_embedded, chunk=pts.shape[0], 
            xyz=xyz,code=code,sigma_only=sigma_only)
    
    sdf = -y
    ibetas = 1/(mlp.beta.abs()+1e-9)
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas))
        
    # get gradient for each size-1 output
    gradients = []
    for i in range(y.shape[-1]):
        y_sub = y [...,i:i+1]
        d_output = torch.ones_like(y_sub, requires_grad=False, device=y.device)
        gradient = torch.autograd.grad(
            outputs=y_sub,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradients.append( gradient[...,None] )
    gradients = torch.cat(gradients,-1) # ...,input-dim, output-dim
    return gradients, sigmas

def compute_gradients_sdf(mlp, embed, pts, sigma_only=False, eps=1e-3):
    """
    Taken from https://github.com/nvlabs/neuralangelo
    """
    pts = pts.detach()

    #mode == "numerical":
    k1 = torch.tensor([1, -1, -1], dtype=pts.dtype, device=pts.device)  # [3]
    k2 = torch.tensor([-1, -1, 1], dtype=pts.dtype, device=pts.device)  # [3]
    k3 = torch.tensor([-1, 1, -1], dtype=pts.dtype, device=pts.device)  # [3]
    k4 = torch.tensor([1, 1, 1], dtype=pts.dtype, device=pts.device)  # [3]
    # sdf1 = fn(pts + k1 * eps)  # [...,1]
    pts_embedded1 = embed(pts + k1 * eps)
    sdf1 = evaluate_mlp(mlp, pts_embedded1, chunk=pts.shape[0], 
            xyz=None,code=None,sigma_only=sigma_only)
    pts_embedded2 = embed(pts + k2 * eps)
    sdf2 = evaluate_mlp(mlp, pts_embedded2, chunk=pts.shape[0], 
            xyz=None,code=None,sigma_only=sigma_only)
    pts_embedded3 = embed(pts + k3 * eps)
    sdf3 = evaluate_mlp(mlp, pts_embedded3, chunk=pts.shape[0], 
            xyz=None,code=None,sigma_only=sigma_only)
    pts_embedded4 = embed(pts + k4 * eps)
    sdf4 = evaluate_mlp(mlp, pts_embedded4, chunk=pts.shape[0], 
            xyz=None,code=None,sigma_only=sigma_only)
    gradient = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
    return gradient

def eikonal_loss(mlp, embed, pts, bound, ppr_eikonal):
    """
    pts: X* backward warped points
    """
    # make it more efficient
    bs = pts.shape[0]
    sample_size = 1000
    if bs>sample_size:
        probs = torch.ones(bs)
        rand_inds = torch.multinomial(probs, sample_size, replacement=False)
        pts = pts[rand_inds]

    pts = pts.view(-1,3).detach()
    nsample = pts.shape[0]
    device = next(mlp.parameters()).device
    bound = torch.Tensor(bound)[None].to(device)

    inbound_idx = ((bound - pts.abs()) > 0).sum(-1) == 3
    pts = pts[inbound_idx]

    pts = pts[None]
    if ppr_eikonal:
        ### ppr eikonal
        g = compute_gradients_sdf(mlp, embed, pts, sigma_only=True)
    else:
        g,sigmas_unit = nerf_gradient(mlp, embed, pts, sigma_only=True)
        g = g[...,0]
    
    grad_norm =  g.norm(2, dim=-1)
    eikonal_loss = (grad_norm - 1) ** 2
    eikonal_loss = eikonal_loss.mean()
    return eikonal_loss
    
def elastic_loss(mlp, embed, xyz, time_embedded):
    xyz = xyz.detach().clone()
    time_embedded = time_embedded.detach().clone()
    g,_ = nerf_gradient(mlp, embed, xyz, use_xyz=mlp.use_xyz,code=time_embedded)
    jacobian = g+torch.eye(3)[None,None].to(g.device)

    sign, log_svals = jacobian.slogdet()
    log_svals = log_svals.clone()
    log_svals[sign<=0] = 0.
    elastic_loss = log_svals**2
    return elastic_loss
    

def bone_density_loss(mlp, embed, bones):
    pts = bones[:,:3] 
    pts_embedded = embed(pts)
    y = evaluate_mlp(mlp, pts_embedded, pts.shape[0], sigma_only=True)
    return bone_density_loss

def visibility_loss(mlp, embed, xyz_pos, w_pos, bound, chunk):
    """
    w_pos: num_points x num_samples, visibility returns from nerf
    bound: scalar, used to sample negative samples
    """
    device = next(mlp.parameters()).device
    xyz_pos = xyz_pos.detach().clone()
    w_pos = w_pos.detach().clone()
    
    # negative examples
    nsample = w_pos.shape[0]*w_pos.shape[1]
    bound = torch.Tensor(bound)[None,None]
    xyz_neg = torch.rand(1,nsample,3)*2*bound-bound
    xyz_neg = xyz_neg.to(device)
    xyz_neg_embedded = embed(xyz_neg)
    vis_neg_pred = evaluate_mlp(mlp, xyz_neg_embedded, chunk=chunk)[...,0]
    vis_loss_neg = -F.logsigmoid(-vis_neg_pred).sum()*0.1/nsample
      
    # positive examples
    xyz_pos_embedded = embed(xyz_pos)
    vis_pos_pred = evaluate_mlp(mlp, xyz_pos_embedded, chunk=chunk)[...,0]
    vis_loss_pos = -(F.logsigmoid(vis_pos_pred) * w_pos).sum()/nsample

    vis_loss = vis_loss_pos + vis_loss_neg
    return vis_loss

def rtk_loss(rtk, rtk_raw, aux_out):
    rot_pred = rtk[:,:3,:3]
    rot_gt = rtk_raw[:,:3,:3]
    rot_loss = rot_angle(rot_pred.matmul(rot_gt.permute(0,2,1))).mean()
    rot_loss = 0.01*rot_loss

    trn_pred = rtk[:,:3,3]
    trn_gt = rtk_raw[:,:3,3]
    trn_loss = (trn_pred - trn_gt).pow(2).sum(-1).mean()
    total_loss = rot_loss + trn_loss
    aux_out['rot_loss'] = rot_loss
    aux_out['trn_loss'] = trn_loss
    return total_loss

def compute_pts_exp(pts_prob, pts):
    """
    pts:      ..., ndepth, 3
    pts_prob: ..., ndepth
    """
    ndepth = pts_prob.shape[-1]
    pts_prob = pts_prob.clone()
    pts_prob = pts_prob.view(-1, ndepth,1)
    pts_prob = pts_prob/(1e-9+pts_prob.sum(1)[:,None])
    pts_exp = (pts * pts_prob).sum(1)
    return pts_exp

def feat_match_loss(nerf_feat, embedding_xyz, feats, pts, pts_prob, bound, use_corr=True,use_ot=False,
        is_training=True):
    """
    feats:    ..., num_feat
    pts:      ..., ndepth, 3
    pts_prob: ..., ndepth
    loss:     ..., 1
    """
    pts = pts.clone()

    base_shape = feats.shape[:-1] # bs, ns
    nfeat =     feats.shape[-1]
    ndepth = pts_prob.shape[-1]
    feats=        feats.view(-1, nfeat)
    pts =           pts.view(-1, ndepth,3)
    
    # part1: compute expected pts
    pts_exp = compute_pts_exp(pts_prob, pts)

    ## part2: matching
    pts_pred, corr_err = feat_match(nerf_feat, embedding_xyz, feats, 
            bound,grid_size=20,use_corr=use_corr,use_ot=use_ot, is_training=is_training)

    # part3: compute loss
    feat_err = (pts_pred - pts_exp).norm(2,-1) # n,ndepth

    # rearrange outputs
    pts_pred  = pts_pred.view(base_shape+(3,))
    pts_exp   = pts_exp .view(base_shape+(3,))
    feat_err = feat_err .view(base_shape+(1,))
    if use_corr:
        corr_err = corr_err.view(base_shape+(1,))
    return pts_pred, pts_exp, feat_err, corr_err

def kp_reproj_loss(pts_pred, xys, models, embedding_xyz, rays, neudbs=True):
    """
    pts_pred,   ...,3
    xys,        ...,2
    out,        ...,1 same as pts_pred
    gcc loss is only used to update root/body pose and skinning weights
    """
    xys = xys.view(-1,1,2)
    xy_reproj = kp_reproj(pts_pred, models, embedding_xyz, rays, neudbs=neudbs) 
    proj_err = (xys - xy_reproj[...,:2]).norm(2,-1)
    proj_err = proj_err.view(pts_pred.shape[:-1]+(1,))
    return proj_err

def kp_reproj(pts_pred, models, embedding_xyz, rays, to_target=False, neudbs=True):
    """
    pts_pred,   ...,3
    out,        ...,1,3 same as pts_pred
    to_target   whether reproject to target frame
    """
    N = pts_pred.view(-1,3).shape[0]
    xyz_coarse_sampled = pts_pred.view(-1,1,3)
    # detach grad since reproj-loss would not benefit feature learning 
    # (due to ambiguity)
    #xyz_coarse_sampled = xyz_coarse_sampled.detach() 

    # TODO wrap flowbw and lbs into the same module
    # TODO include loss for flowbw
    if to_target:  rtk_vec = rays['rtk_vec_target']
    else:          rtk_vec = rays['rtk_vec']
    rtk_vec = rtk_vec.view(N,-1) # bs, ns, 21
    if 'bones' in models.keys():
        if to_target:    bone_rts_fw = rays['bone_rts_target']
        else:            bone_rts_fw = rays['bone_rts']
        bone_rts_fw = bone_rts_fw.view(N,-1) # bs, ns,-1
        if 'nerf_skin' in models.keys():
            nerf_skin = models['nerf_skin']
        else: nerf_skin = None
        bones = models['bones_rst']
        skin_aux = models['skin_aux']
        rest_pose_code = models['rest_pose_code']
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones.device))
        skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, bones,
                                  rest_pose_code, nerf_skin, skin_aux=skin_aux)
        if neudbs:
            xyz_coarse_sampled,_,_ = neu_dbs(bones, bone_rts_fw,
                          skin_forward, xyz_coarse_sampled, backward=False)
        else:
            xyz_coarse_sampled,_ = lbs(bones, bone_rts_fw,
                            skin_forward, xyz_coarse_sampled, backward=False)

    Rmat = rtk_vec[:,0:9]  .view(N,1,3,3)
    Tmat = rtk_vec[:,9:12] .view(N,1,3)
    Kinv = rtk_vec[:,12:21].view(N,1,3,3)
    K = mat2K(Kmatinv(Kinv))

    xyz_coarse_sampled = obj_to_cam( xyz_coarse_sampled, Rmat, Tmat) 
    xyz_coarse_sampled = pinhole_cam(xyz_coarse_sampled,K)

    xy_coarse_sampled = xyz_coarse_sampled[...,:2]
    return xy_coarse_sampled
    
    
def feat_match(nerf_feat, embedding_xyz, feats, bound, 
        grid_size=20,use_corr=True,use_ot=False, is_training=True, init_pts=None, rt_entropy=False):
    """
    feats:    -1, num_feat
    """
    if is_training: 
        chunk_pts = 8*1024
    else:
        chunk_pts = 1024
    chunk_pix = 4096
    nsample,_ = feats.shape
    device = feats.device
    feats = F.normalize(feats,2,-1)
    
    # sample model on a regular 3d grid, and correlate with feature, nkxkxk
    #p1d = np.linspace(-bound, bound, grid_size).astype(np.float32)
    #query_yxz = np.stack(np.meshgrid(p1d, p1d, p1d), -1)  # (y,x,z)
    pxd = np.linspace(-bound[0], bound[0], grid_size).astype(np.float32)
    pyd = np.linspace(-bound[1], bound[1], grid_size).astype(np.float32)
    pzd = np.linspace(-bound[2], bound[2], grid_size).astype(np.float32)
    query_yxz = np.stack(np.meshgrid(pyd, pxd, pzd), -1)  # (y,x,z)
    query_yxz = torch.Tensor(query_yxz).to(device).view(-1, 3)
    query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)

    if init_pts is not None:
        query_xyz = query_xyz[None] + init_pts[:,None]
    else:
        # N x Ns x 3
        query_xyz = query_xyz[None]

    # inject some noise at training time
    if is_training and init_pts is None:
        bound = torch.Tensor(bound)[None,None].to(device)
        query_xyz = query_xyz + torch.randn_like(query_xyz) * bound * 0.05

    cost_vol = []
    for i in range(0,grid_size**3,chunk_pts):
        if init_pts is None:
            query_xyz_chunk = query_xyz[0,i:i+chunk_pts]
            xyz_embedded = embedding_xyz(query_xyz_chunk)[:,None] # (N,1,...)
            vol_feat_subchunk = evaluate_mlp(nerf_feat, xyz_embedded)[:,0] # (chunk, num_feat)
            # normalize vol feat
            vol_feat_subchunk = F.normalize(vol_feat_subchunk,2,-1)[None]
            # print("check, check, check, check!!!!!!!!")
            # print(vol_feat_subchunk.shape)

        cost_chunk = []
        for j in range(0,nsample,chunk_pix):
            feats_chunk = feats[j:j+chunk_pix] # (chunk pix, num_feat)
     
            if init_pts is not None:
                # only query 3d grid according to each px when they are diff
                # vol feature
                query_xyz_chunk = query_xyz[j:j+chunk_pix,i:i+chunk_pts].clone()
                xyz_embedded = embedding_xyz(query_xyz_chunk)
                vol_feat_subchunk = evaluate_mlp(nerf_feat, xyz_embedded)
                # normalize vol feat
                vol_feat_subchunk = F.normalize(vol_feat_subchunk,2,-1)              
            # cpix, cpts
           # distance metric
            if use_ot:
                cost_subchunk = (vol_feat_subchunk * \
                    feats_chunk[:,None]).sum(-1)
            else:
                cost_subchunk = (vol_feat_subchunk * \
                        feats_chunk[:,None]).sum(-1) * (nerf_feat.beta.abs()+1e-9)
            cost_chunk.append(cost_subchunk)
        cost_chunk = torch.cat(cost_chunk,0) # (nsample, cpts)
        cost_vol.append(cost_chunk)
    cost_vol = torch.cat(cost_vol,-1) # (nsample, k**3)

    if use_ot:
        cost_vol = cost_vol[None]
        # Optimal Transport
        K = torch.exp(-(1.0 - cost_vol) / 0.03)

        # Init. of Sinkhorn algorithm
        power = 1#gamma / (gamma + epsilon)
        a = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=vol_feat_subchunk.device, dtype=vol_feat_subchunk.dtype
            )
            / K.shape[1]
        )
        prob1 = (
            torch.ones(
                (K.shape[0], K.shape[1], 1), device=vol_feat_subchunk.device, dtype=vol_feat_subchunk.dtype
            )
            / K.shape[1]
        )
        prob2 = (
            torch.ones(
                (K.shape[0], K.shape[2], 1), device=feats_chunk.device, dtype=feats_chunk.dtype
            )
            / K.shape[2]
        )

        # Sinkhorn algorithm
        for _ in range(20):
            # Update b
            KTa = torch.bmm(K.transpose(1, 2), a)
            b = torch.pow(prob2 / (KTa + 1e-8), power)
            # Update a
            Kb = torch.bmm(K, b)
            a = torch.pow(prob1 / (Kb + 1e-8), power)

        # Optimal matching matrix Tm
        T_m = torch.mul(torch.mul(a, K), b.transpose(1, 2))
        T_m_norm = T_m / torch.sum(T_m, dim=2, keepdim=True)
        prob_vol = T_m_norm[0]
    else:
        prob_vol = cost_vol.softmax(-1)

    #calculate backward correspondence loss
    if use_corr:
        T_T = torch.matmul(prob_vol, prob_vol.transpose(1,0))  # 
        I = torch.eye(prob_vol.shape[0]).to(T_T.device)
        # corr_err = torch.mean((T_T - I)**2) 
        corr_err = (T_T - I).norm(2,-1)
    else:
        corr_err = 0

    # regress to the true location, n,3
    if not is_training: torch.cuda.empty_cache()
    # n, ns, 1 * n, ns, 3
    pts_pred = (prob_vol[...,None] * query_xyz).sum(1)
    if rt_entropy:
        # compute normalized entropy
        match_unc = (-prob_vol * prob_vol.clamp(1e-9,1-1e-9).log()).sum(1)[:,None]
        match_unc = match_unc/np.log(grid_size**3)
        return pts_pred, match_unc, corr_err
    else:
        return pts_pred, corr_err

def grad_update_bone(bones,embedding_xyz, nerf_vis, learning_rate):
    """
    #TODO need to update bones locally
    """
    device = bones.device
    bones_data = bones.data.detach()
    bones_data.requires_grad_(True)
    bone_xyz_embed = embedding_xyz(bones_data[:,None,:3])
    sdf_at_bone = evaluate_mlp(nerf_vis, bone_xyz_embed)
    bone_loc_loss = F.relu(-sdf_at_bone).mean()
    
    # compute gradient wrt bones
    d_output = torch.ones_like(bone_loc_loss, requires_grad=False, device=device)
    gradient = torch.autograd.grad(
        outputs=bone_loc_loss,
        inputs=bones_data,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    bones.data = bones.data-gradient*learning_rate

    return bone_loc_loss

def loss_filter_line(sil_err, errid, frameid, sil_loss_samp, img_size, scale_factor=10):
    """
    sil_err: Tx512
    errid: N
    """
    sil_loss_samp = sil_loss_samp.detach().cpu().numpy().reshape(-1)
    sil_err[errid] = sil_loss_samp
    sil_err = sil_err.reshape(-1,img_size)
    sil_err = sil_err.sum(-1) / (1e-9+(sil_err>0).astype(float).sum(-1))
    sil_err_med = np.median(sil_err[sil_err>0])
    invalid_frame = sil_err > sil_err_med*scale_factor
    invalid_idx = invalid_frame[frameid]
    sil_err[:] = 0
    return invalid_idx

def loss_filter(g_floerr, flo_loss_samp, sil_at_samp_flo, scale_factor=10):
    """
    g_floerr:       T,
    flo_loss_samp:  bs,N,1
    sil_at_samp_flo:bs,N,1
    """
    bs = sil_at_samp_flo.shape[0] 
    # find history meidan
    g_floerr = g_floerr[g_floerr>0]

    # tb updated as history value
    #flo_err = []
    #for i in range(bs):
    #    flo_err_sub =flo_loss_samp[i][sil_at_samp_flo[i]]
    #    if len(flo_err_sub) >0:
    #        #flo_err_sub = flo_err_sub.median().detach().cpu().numpy()
    #        flo_err_sub = flo_err_sub.mean().detach().cpu().numpy()
    #    else: 
    #        flo_err_sub = 0
    #    flo_err.append(flo_err_sub)
    #flo_err = np.stack(flo_err)
    
    # vectorized version but uses mean to update
    flo_err = (flo_loss_samp * sil_at_samp_flo).sum(1) /\
              (1e-9+sil_at_samp_flo.sum(1)) # bs, N, 1
    flo_err = flo_err.detach().cpu().numpy()[...,0]

    # find invalid idx
    invalid_idx = flo_err > np.median(g_floerr)*scale_factor
    return flo_err, invalid_idx


def compute_xyz_wt_loss(gt_list, curr_list):
    loss = []
    for i in range(len(gt_list)):
        loss.append( (gt_list[i].detach() - curr_list[i]).pow(2).mean() )
    loss = torch.stack(loss).mean()
    return loss

def compute_root_sm_2nd_loss(rtk_all, data_offset):
    """
    2nd order loss
    """
    rot_sm_loss = []
    trn_sm_loss = []
    for didx in range(len(data_offset)-1):
        stt_idx = data_offset[didx]
        end_idx = data_offset[didx+1]

        stt_rtk = rtk_all[stt_idx:end_idx-2]
        mid_rtk = rtk_all[stt_idx+1:end_idx-1]
        end_rtk = rtk_all[stt_idx+2:end_idx]

        rot_sub1 = stt_rtk[:,:3,:3].matmul(mid_rtk[:,:3,:3].permute(0,2,1))
        rot_sub2 = mid_rtk[:,:3,:3].matmul(end_rtk[:,:3,:3].permute(0,2,1))

        trn_sub1 = stt_rtk[:,:3,3] - mid_rtk[:,:3,3]
        trn_sub2 = mid_rtk[:,:3,3] - end_rtk[:,:3,3]

        rot_sm_sub = rot_sub1.matmul(rot_sub2.permute(0,2,1))
        trn_sm_sub = trn_sub1 - trn_sub2
        
        rot_sm_loss.append(rot_sm_sub)
        trn_sm_loss.append(trn_sm_sub)
    rot_sm_loss = torch.cat(rot_sm_loss,0)
    rot_sm_loss = rot_angle(rot_sm_loss).mean()*1e-1
    trn_sm_loss = torch.cat(trn_sm_loss,0)
    trn_sm_loss = trn_sm_loss.norm(2,-1).mean()
    root_sm_loss = rot_sm_loss + trn_sm_loss 
    root_sm_loss = root_sm_loss * 0.1
    return root_sm_loss


def compute_root_sm_loss(rtk_all, data_offset):
    rot_sm_loss = []
    trans_sm_loss = []
    for didx in range(len(data_offset)-1):
        stt_idx = data_offset[didx]
        end_idx = data_offset[didx+1]
        rot_sm_sub = rtk_all[stt_idx:end_idx-1,:3,:3].matmul(
                      rtk_all[stt_idx+1:end_idx,:3,:3].permute(0,2,1))
        trans_sm_sub =  rtk_all[stt_idx:end_idx-1,:3,3] - \
                        rtk_all[stt_idx+1:end_idx,:3,3]
        rot_sm_loss.append(rot_sm_sub)
        trans_sm_loss.append(trans_sm_sub)
    rot_sm_loss = torch.cat(rot_sm_loss,0)
    rot_sm_loss = rot_angle(rot_sm_loss).mean()*1e-3
    trans_sm_loss = torch.cat(trans_sm_loss,0)
    trans_sm_loss = trans_sm_loss.norm(2,-1).mean()*0.1
    root_sm_loss = rot_sm_loss + trans_sm_loss 
    return root_sm_loss


def shape_init_loss(pts, faces,  mlp, embed, bound_factor, use_ellips=True):
    # compute sdf loss wrt to a mesh
    # construct mesh
    mesh = trimesh.Trimesh(pts.cpu(), faces=faces.cpu())
    device = next(mlp.parameters()).device

    # Sample points
    nsample =10000
    obj_bound = pts.abs().max(0)[0][None,None]
    bound = obj_bound * bound_factor
    pts_samp = torch.rand(1,nsample,3).to(device)*2*bound-bound

    # outside: positive
    if use_ellips:
        # signed distance to a ellipsoid
        dis = (pts_samp/obj_bound).pow(2).sum(2).view(-1)
        dis = torch.sqrt(dis)
        dis = dis  - 1 
        dis = dis * obj_bound.mean()
    else:
        # signed distance to a sphere
        dis = (pts_samp).pow(2).sum(2).view(-1)
        dis = torch.sqrt(dis)
        dis = dis  - obj_bound.min()

    # compute sdf
    pts_embedded = embed(pts_samp)
    y = evaluate_mlp(mlp, pts_embedded, chunk=pts_samp.shape[0], 
            xyz=None,code=None,sigma_only=True)
    
    sdf = -y.view(-1) # positive: outside
    shape_loss = (sdf - dis).pow(2).mean()
    return shape_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, stride=None):
    mu1 = F.conv2d(img1, window, padding = (window_size-1)//2, groups = channel, stride=stride)
    img2 = img2.to(window.dtype)
    mu2 = F.conv2d(img2, window, padding = (window_size-1)//2, groups = channel, stride=stride)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = (window_size-1)//2, groups = channel, stride=stride) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, stride=self.stride)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper  
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10, patch_height=64, patch_width=64):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)
    def forward(self, src_vec, tar_vec, mask):
        loss = 0.0
        index_list = []
        src_vec *=mask
        tar_vec *=mask
        src_vec = src_vec.reshape(-1, 3)
        tar_vec = tar_vec.reshape(-1, 3)
        n = src_vec.size(0)
        if n >= self.patch_height*self.patch_width:
            src_vec = src_vec[:self.patch_height*self.patch_width, :]
            tar_vec = tar_vec[:self.patch_height*self.patch_width, :]
        else:
            repetitions = -(-self.patch_height*self.patch_width // n)  # Equivalent to ceil(target_shape[0] / n)
            src_vec = src_vec.repeat(repetitions, 1)
            src_vec = src_vec[:self.patch_height*self.patch_width, :]
            tar_vec = tar_vec.repeat(repetitions, 1)
            tar_vec = tar_vec[:self.patch_height*self.patch_width, :]
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        # import pdb as p
        # p.set_trace()
        # print(tar_all.dtype)
        # print(src_all.dtype)
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.patch_height, self.patch_width * self.repeat_time)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss
