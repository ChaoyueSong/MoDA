# Modified from https://github.com/facebookresearch/banmo

# adopted from nerf-pl
import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d import transforms
import time

from nnutils.geom_utils import lbs, neu_dbs, Kmatinv, mat2K, pinhole_cam, obj_to_cam,\
                               vec_to_sim3, rtmat_invert, rot_angle, mlp_skinning,\
                               bone_transform, skinning, vrender_flo, \
                               gauss_mlp_skinning, diff_flo
from nnutils.loss_utils import elastic_loss, visibility_loss, feat_match_loss,\
                                kp_reproj_loss, compute_pts_exp, kp_reproj, evaluate_mlp, S3IM

def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                chunk=1024*32,
                obj_bound=None,
                use_fine=False,
                img_size=None,
                progress=None,
                opts=None,
                render_vis=False,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        chunk: the chunk size in batched inference

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    if use_fine: N_samples = N_samples//2 # use half samples to importance sample

    # Extract models from lists
    embedding_xyz = embeddings['xyz']
    embedding_dir = embeddings['dir']

    # Decompose the inputs
    rays_o = rays['rays_o']
    rays_d = rays['rays_d']  # both (N_rays, 3)
    near = rays['near']
    far = rays['far']  # both (N_rays, 1)
    N_rays = rays_d.shape[0]
 
    # Embed direction
    rays_d_norm = rays_d / rays_d.norm(2,-1)[:,None]
    dir_embedded = embedding_dir(rays_d_norm) # (N_rays, embed_dir_channels)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays_d.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays_d.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # zvals are not optimized
    # produce points in the root body space
    
    xyz_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)
    
    if use_fine: # sample points for fine model
        # output: 
        #  loss:   'img_coarse', 'sil_coarse', 'feat_err', 'proj_err' 
        #               'vis_loss', 'flo/fdp_coarse', 'flo/fdp_valid',  
        #  not loss:   'depth_rnd', 'pts_pred', 'pts_exp'
        with torch.no_grad():
            _, weights_coarse = inference_deform(xyz_sampled, rays, models, 
                              chunk, N_samples,
                              N_rays, embedding_xyz, rays_d, noise_std,
                              obj_bound, dir_embedded, z_vals,
                              img_size, progress,opts,fine_iter=False)

        # reset N_importance
        N_importance = N_samples
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) 
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
        N_samples = N_samples + N_importance # get back to original # of samples
    
    result, _ = inference_deform(xyz_sampled, rays, models, 
                          chunk, N_samples,
                          N_rays, embedding_xyz, rays_d, noise_std,
                          obj_bound, dir_embedded, z_vals,
                          img_size, progress,opts,render_vis=render_vis)

    return result
    
def inference(models, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code=None, appearance_code=None, weights_only=False, clip_bound = None, vis_pred=None, scale_rgb=1.3, rgb_filter=False):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine)
        embedding_xyz: embedding module for xyz
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        dir_embedded: (N_rays, embed_dir_channels) embedded directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        rgb_final: (N_rays, 3) the final rgb image
        depth_final: (N_rays) depth map
        weights: (N_rays, N_samples_): weights of each sample
    """
    nerf_sdf = models['coarse']
    N_samples_ = xyz_.shape[1]
    # Embed directions
    xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
    if not weights_only:
        dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                       # (N_rays*N_samples_, embed_dir_channels)

    # Perform model inference to get rgb and raw sigma
    chunk_size=4096
    B = xyz_.shape[0]
    xyz_input = xyz_.view(N_rays,N_samples,3)
    out = evaluate_mlp(nerf_sdf, xyz_input, 
            embed_xyz = embedding_xyz,
            dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
            code=env_code, appearance_code=appearance_code,
            chunk=chunk_size, sigma_only=weights_only, use_semantic=False).view(B,-1)

    rgbsigma = out.view(N_rays, N_samples_, 4)
    rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
    sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)
    # sem_logits = rgbsigma[..., 4] # (N_rays, N_samples_)

    # add semantic
    # scale_rgb = 2 
    semantic = scale_rgb*torch.sigmoid(-10 * sigmas)   #sigmoid=10
    
    if 'nerf_feat' in models.keys():
        nerf_feat = models['nerf_feat']
        feat = evaluate_mlp(nerf_feat, xyz_input,
            embed_xyz = embedding_xyz,
            chunk=chunk_size).view(N_rays,N_samples_,-1)
    else:
        feat = torch.zeros_like(rgbs)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    # a hacky way to ensures prob. sum up to 1     
    # while the prob. of last bin does not correspond with the values
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by the formula (3)
    sigmas = sigmas+noise
    #sigmas = F.softplus(sigmas)
    #sigmas = torch.relu(sigmas)
    ibetas = 1/(nerf_sdf.beta.abs()+1e-9)
    #ibetas = 100
    sdf = -sigmas
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
    # alternative: 
    #sigmas = F.sigmoid(-sdf*ibetas)
    sigmas = sigmas * ibetas

    alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i

    #set out-of-bound and nonvisible alphas to zero
    if clip_bound is not None:
        clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
        oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
        alphas[oob]=0
    if vis_pred is not None:
        alphas[vis_pred<0.5] = 0

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]
    weights = alphas * alpha_prod # (N_rays, N_samples_)
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically  equals 1
 
    visibility = alpha_prod.detach() # 1 q_0 q_j-1
    sem_weights = weights[:, :-1]*semantic[:, :-1]
    norm_weight = torch.clamp(sem_weights.sum(1)+1e-9,0.3,1)
    # compute final weighted outputs
    # rgb_filter=False
    if rgb_filter:
        rgb_final = torch.sum((sem_weights).unsqueeze(-1)*rgbs[:,:-1,:], -2) # (N_rays, 3)   (3072*128*1)*(3072,128,3) 
    else:
        rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2)
    feat_final = torch.sum(weights.unsqueeze(-1)*feat, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)
    sil_final = torch.sum(weights[:, :-1], -1)
    
    return rgb_final, feat_final, depth_final, weights, visibility, sil_final
    
def inference_deform(xyz_coarse_sampled, rays, models, chunk, N_samples,
                         N_rays, embedding_xyz, rays_d, noise_std,
                         obj_bound, dir_embedded, z_vals,
                         img_size, progress, opts, fine_iter=True, 
                         render_vis=False):
    """
    fine_iter: whether to render loss-related terms
    render_vis: used for novel view synthesis
    """
    is_training = models['coarse'].training
    xys = rays['xys']

    # root space point correspondence in t2
    if opts.dist_corresp:
        xyz_coarse_target = xyz_coarse_sampled.clone()
        xyz_coarse_dentrg = xyz_coarse_sampled.clone()
    xyz_coarse_frame  = xyz_coarse_sampled.clone()
    result= {}
    # free deform
    if 'flowbw' in models.keys():
        model_flowbw = models['flowbw']
        model_flowfw = models['flowfw']
        time_embedded = rays['time_embedded'][:,None]
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, 
                             chunk=chunk//N_samples, xyz=xyz_coarse_sampled, code=time_embedded)
        xyz_coarse_sampled=xyz_coarse_sampled + flow_bw
       
        if fine_iter:
            # cycle loss (in the joint canonical space)
            xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
            flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                  chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded)
            frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
            # rigidity loss
            frame_disp3d = flow_fw.norm(2,-1)

            if "time_embedded_target" in rays.keys():
                time_embedded_target = rays['time_embedded_target'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_target)
                xyz_coarse_target=xyz_coarse_sampled + flow_fw
            
            if "time_embedded_dentrg" in rays.keys():
                time_embedded_dentrg = rays['time_embedded_dentrg'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_dentrg)
                xyz_coarse_dentrg=xyz_coarse_sampled + flow_fw


    elif 'bones' in models.keys():
        bones_rst = models['bones_rst']
        bone_rts_fw = rays['bone_rts']
        skin_aux = models['skin_aux']
        rest_pose_code =  models['rest_pose_code']
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        
        if 'nerf_skin' in models.keys():
            # compute delta skinning weights of bs, N, B
            nerf_skin = models['nerf_skin'] 
        else:
            nerf_skin = None
        time_embedded = rays['time_embedded'][:,None]
  
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, opts.neudbs, is_vec=True)
        skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                    bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)

        # backward skinning
        if 'nerf_dis' in models.keys():
            nerf_dis = models['nerf_dis'] 
        else:
            nerf_dis = None
        if opts.lbs:
            xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                    bone_rts_fw, 
                                                    skin_backward,
                                                    xyz_coarse_sampled
                                                    ) #add dis
        elif opts.neudbs:
            xyz_coarse_sampled, bones_dfm, xyz_dis = neu_dbs(bones_rst, 
                                                    bone_rts_fw, 
                                                    skin_backward,
                                                    xyz_coarse_sampled, nerf_dis, embedding_xyz, time_embedded
                                                    ) #add dis

        if 'nerf_dis' in models.keys():
            result['dis_reg'] = torch.norm(xyz_dis, dim=2, keepdim = False)

        if fine_iter:
            #if opts.dist_corresp:
            skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                        bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)

            # cycle loss (in the joint canonical space)
            if opts.lbs:
                xyz_coarse_frame_cyc,_= lbs(bones_rst, bone_rts_fw,
                                skin_forward, xyz_coarse_sampled, backward=False)
            elif opts.neudbs:
                xyz_coarse_frame_cyc,_, xyz_dis_forward = neu_dbs(bones_rst, bone_rts_fw,
                                skin_forward, xyz_coarse_sampled, nerf_dis, embedding_xyz, rest_pose_code, backward=False)
                
            frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)
            if 'nerf_dis' in models.keys():
                result['dis_reg_forward'] = torch.norm(xyz_dis_forward, dim=2, keepdim=False)
            
            if opts.dist_corresp and 'bone_rts_target' in rays.keys():
                bone_rts_target = rays['bone_rts_target']
                if opts.lbs:
                    xyz_coarse_target,_= lbs(bones_rst, bone_rts_target, 
                                    skin_forward, xyz_coarse_sampled, backward=False)
                elif opts.neudbs:
                    xyz_coarse_target,_, _= neu_dbs(bones_rst, bone_rts_target, 
                                    skin_forward, xyz_coarse_sampled, nerf_dis, embedding_xyz, rest_pose_code, backward=False)
            if opts.dist_corresp and 'bone_rts_dentrg' in rays.keys():
                bone_rts_dentrg = rays['bone_rts_dentrg']
                if opts.lbs:
                    xyz_coarse_dentrg,_ = lbs(bones_rst, bone_rts_dentrg, 
                                    skin_forward, xyz_coarse_sampled, backward=False)
                elif opts.neudbs:
                    xyz_coarse_dentrg,_,_ = neu_dbs(bones_rst, bone_rts_dentrg, 
                                    skin_forward, xyz_coarse_sampled, nerf_dis, embedding_xyz, rest_pose_code, backward=False)

    # nerf shape/rgb
    model_coarse = models['coarse']
    if 'env_code' in rays.keys():
        env_code = rays['env_code']
    else:
        env_code = None

    if 'appearance_code' in rays.keys():
        appearance_code = rays['appearance_code']
    else:
        appearance_code = None

    # set out of bounds weights to zero
    if render_vis: 
        clip_bound = obj_bound
        xyz_embedded = embedding_xyz(xyz_coarse_sampled)
        vis_pred = evaluate_mlp(models['nerf_vis'], 
                               xyz_embedded, chunk=chunk)[...,0].sigmoid()
    else:
        clip_bound = None
        vis_pred = None


    if opts.symm_shape:
        ##TODO set to x-symmetric here
        symm_ratio = 0.5
        xyz_x = xyz_coarse_sampled[...,:1].clone()
        symm_mask = torch.rand_like(xyz_x) < symm_ratio
        xyz_x[symm_mask] = -xyz_x[symm_mask]
        xyz_input = torch.cat([xyz_x, xyz_coarse_sampled[...,1:3]],-1)
    else:
        xyz_input = xyz_coarse_sampled

    rgb_coarse, feat_rnd, depth_rnd, weights_coarse, vis_coarse, sil_coarse = \
        inference(models, embedding_xyz, xyz_input, rays_d,
                dir_embedded, z_vals, N_rays, N_samples, chunk, noise_std,
                weights_only=False, env_code=env_code, appearance_code=appearance_code,
                clip_bound=clip_bound, vis_pred=vis_pred, scale_rgb=opts.scale_rgb, rgb_filter=opts.rgb_filter)
    sil_coarse =  weights_coarse[:,:-1].sum(1)
   
    result['img_coarse'] = rgb_coarse
    result['depth_rnd'] = depth_rnd
    result['sil_coarse'] = sil_coarse
   
    # render visibility scores
    if render_vis:
        result['vis_pred'] = (vis_pred * weights_coarse).sum(-1)

    if fine_iter:
        if opts.use_corresp:
            # for flow rendering
            pts_exp = compute_pts_exp(weights_coarse, xyz_coarse_sampled)
            pts_target = kp_reproj(pts_exp, models, embedding_xyz, rays, 
                                to_target=True, neudbs=opts.neudbs) # N,1,2
        # viser feature matching
        if 'feats_at_samp' in rays.keys():
            feats_at_samp = rays['feats_at_samp']
            nerf_feat = models['nerf_feat']
            xyz_coarse_sampled_feat = xyz_coarse_sampled
            weights_coarse_feat = weights_coarse
            pts_pred, pts_exp, feat_err, corr_err = feat_match_loss(nerf_feat, embedding_xyz,
                       feats_at_samp, xyz_coarse_sampled_feat, weights_coarse_feat,
                       obj_bound, opts.use_corr, opts.use_ot, is_training=is_training)


            # 3d-2d projection
            proj_err = kp_reproj_loss(pts_pred, xys, models, 
                    embedding_xyz, rays, neudbs=opts.neudbs)
            proj_err = proj_err/img_size * 2
            
            result['pts_pred'] = pts_pred
            result['pts_exp']  = pts_exp
            result['feat_err'] = feat_err # will be used as loss
            if opts.use_corr:
                result['corr_err'] = corr_err # will be used as loss
            result['proj_err'] = proj_err # will be used as loss

        if opts.dist_corresp and 'rtk_vec_target' in rays.keys():
            # compute correspondence: root space to target view space
            # RT: root space to camera space
            rtk_vec_target =  rays['rtk_vec_target']
            Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat, Tmat) 
            xyz_coarse_target = pinhole_cam(xyz_coarse_target,K)

        if opts.dist_corresp and 'rtk_vec_dentrg' in rays.keys():
            # compute correspondence: root space to dentrg view space
            # RT: root space to camera space
            rtk_vec_dentrg =  rays['rtk_vec_dentrg']
            Rmat = rtk_vec_dentrg[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_dentrg[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_dentrg[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_dentrg = obj_to_cam(xyz_coarse_dentrg, Rmat, Tmat) 
            xyz_coarse_dentrg = pinhole_cam(xyz_coarse_dentrg,K)
        
        # raw 3d points for visualization
        result['xyz_camera_vis']   = xyz_coarse_frame 
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            result['xyz_canonical_vis']   = xyz_coarse_sampled
        if 'feats_at_samp' in rays.keys():
            result['pts_exp_vis']   = pts_exp
            result['pts_pred_vis']   = pts_pred
            
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            # cycle loss (in the joint canonical space)
            result['frame_cyc_dis'] = (frame_cyc_dis * weights_coarse.detach()).sum(-1)

        if is_training and 'nerf_vis' in models.keys():
            result['vis_loss'] = visibility_loss(models['nerf_vis'], embedding_xyz,
                            xyz_coarse_sampled, vis_coarse, obj_bound, chunk)

        # render flow 
        if 'rtk_vec_target' in rays.keys():
            if opts.dist_corresp:
                flo_coarse, flo_valid = vrender_flo(weights_coarse, xyz_coarse_target,
                                                    xys, img_size)
            else:
                flo_coarse = diff_flo(pts_target, xys, img_size)
                flo_valid = torch.ones_like(flo_coarse[...,:1])

            result['flo_coarse'] = flo_coarse
            result['flo_valid'] = flo_valid

        if 'rtk_vec_dentrg' in rays.keys():
            if opts.dist_corresp:
                fdp_coarse, fdp_valid = vrender_flo(weights_coarse, 
                                                    xyz_coarse_dentrg, xys, img_size)
            else:
                fdp_coarse = diff_flo(pts_dentrg, xys, img_size)
                fdp_valid = torch.ones_like(fdp_coarse[...,:1])
            result['fdp_coarse'] = fdp_coarse
            result['fdp_valid'] = fdp_valid

        if 'nerf_unc' in models.keys():
            # xys: bs,nsample,2
            # t: bs
            nerf_unc = models['nerf_unc']
            ts = rays['ts']
            vid_code = rays['vid_code']

            # change according to K
            xysn = rays['xysn']
            xyt = torch.cat([xysn, ts],-1)
            xyt_embedded = embedding_xyz(xyt)
            xyt_code = torch.cat([xyt_embedded, vid_code],-1)
            unc_pred = nerf_unc(xyt_code)
            #TODO add activation function
            #unc_pred = F.softplus(unc_pred)
            result['unc_pred'] = unc_pred
        
        if 'img_at_samp' in rays.keys():
            # compute other losses
            img_at_samp = rays['img_at_samp']
            sil_at_samp = rays['sil_at_samp']
            vis_at_samp = rays['vis_at_samp']
            flo_at_samp = rays['flo_at_samp']
            cfd_at_samp = rays['cfd_at_samp']

            # img loss
            img_loss_samp = (rgb_coarse - img_at_samp).pow(2).mean(-1)[...,None] # rgb_coarse(24576,3); img_loss_samp(24576,1)
            if opts.s3im_loss:
                s3im_func = S3IM(kernel_size=4, stride=4, repeat_time=10, patch_height=32, patch_width=32).cuda()
                # random_indices = torch.randperm(rgb_coarse.shape[0])[:4096]
                # print(rgb_coarse.shape)
                s3im_loss = s3im_func(rgb_coarse, img_at_samp, sil_at_samp)
            
            # sil loss, weight sil loss based on # points
            if is_training and sil_at_samp.sum()>0 and (1-sil_at_samp).sum()>0:
                pos_wt = vis_at_samp.sum()/   sil_at_samp[vis_at_samp>0].sum()
                neg_wt = vis_at_samp.sum()/(1-sil_at_samp[vis_at_samp>0]).sum()
                sil_balance_wt = 0.5*pos_wt*sil_at_samp + 0.5*neg_wt*(1-sil_at_samp)
            else: sil_balance_wt = 1

            CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)
            crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label) 
            # sil_loss_samp = crossentropy_loss(sil_coarse[...,None], sil_at_samp) * sil_balance_wt
            sil_loss_samp = (sil_coarse[...,None] - sil_at_samp).pow(2) * sil_balance_wt
            # sil_loss_samp = F.binary_cross_entropy(sil_coarse[...,None], sil_at_samp) * sil_balance_wt *0.2
            sil_loss_samp = sil_loss_samp * vis_at_samp 
               
            # flo loss, confidence weighting: 30x normalized distance - 0.1x pixel error
            flo_loss_samp = (flo_coarse - flo_at_samp).pow(2).sum(-1)
            # hard-threshold cycle error
            sil_at_samp_flo = (sil_at_samp>0)\
                     & (flo_valid==1)
            sil_at_samp_flo[cfd_at_samp==0] = False 
            if sil_at_samp_flo.sum()>0:
                cfd_at_samp = cfd_at_samp / cfd_at_samp[sil_at_samp_flo].mean()
            flo_loss_samp = flo_loss_samp[...,None] * cfd_at_samp
       
            result['img_at_samp']   = img_at_samp
            result['sil_at_samp']   = sil_at_samp
            result['vis_at_samp']   = vis_at_samp
            result['sil_at_samp_flo']   = sil_at_samp_flo
            result['flo_at_samp']   = flo_at_samp
            result['img_loss_samp'] = img_loss_samp 
            result['sil_loss_samp'] = sil_loss_samp
            result['flo_loss_samp'] = flo_loss_samp
            if opts.s3im_loss:
                result['s3im_loss'] = s3im_loss
    
            # exclude error outside mask
            result['img_loss_samp']*=sil_at_samp
            result['flo_loss_samp']*=sil_at_samp

        if 'feats_at_samp' in rays.keys():
            # feat loss
            feats_at_samp=rays['feats_at_samp']
            feat_rnd = F.normalize(feat_rnd, 2,-1)
            frnd_loss_samp = (feat_rnd - feats_at_samp).pow(2).mean(-1)
            result['frnd_loss_samp'] = frnd_loss_samp * sil_at_samp[...,0]
    return result, weights_coarse


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


