"""Motion corroborates identity; it never supplies a missing light or color."""
import math
import cv2
import numpy as np
from arm_scene_a04 import box_iou,frame_identity


class HeadContinuity:
    def __init__(self,max_gap_ms=500.):
        self.previous=None
        self.serial=0
        self.max_gap_ms=max_gap_ms

    @staticmethod
    def predict_boxes(previous,current_gray,current_boxes):
        gray=previous['gray'];predicted=[];evidence=[]
        h,w=gray.shape
        old=np.asarray(previous['boxes'],np.float32);new=np.asarray(current_boxes,np.float32)
        initial_shift=((new[:,:2]+new[:,2:])/2).mean(0)-((old[:,:2]+old[:,2:])/2).mean(0)
        for box in previous['boxes']:
            a,b,c,d=map(float,box);dx=(c-a)*.6;dy=(d-b)*.8
            left,top=max(0,int(a-dx)),max(0,int(b-dy))
            right,bottom=min(w,int(math.ceil(c+dx))),min(h,int(math.ceil(d+dy)))
            points=cv2.goodFeaturesToTrack(gray[top:bottom,left:right],maxCorners=24,
                                          qualityLevel=.01,minDistance=2,blockSize=3)
            reason='insufficient_current_motion_support';candidate=list(box);inliers=0
            if points is not None and len(points)>=4:
                global_points=points+np.float32([left,top])
                # Build small paired image pyramids around this head. Rebuilding
                # two full-frame pyramids per lamp consumed the freshness budget.
                mx,my=max(32.,1.5*(c-a)),max(32.,2.5*(d-b))
                xl,yt=max(0,int(a-mx)),max(0,int(b-my))
                xr,yb=min(w,int(math.ceil(c+mx))),min(h,int(math.ceil(d+my)))
                offset=np.float32([xl,yt]);points=global_points-offset
                old_roi=np.ascontiguousarray(gray[yt:yb,xl:xr])
                new_roi=np.ascontiguousarray(current_gray[yt:yb,xl:xr])
                initial=np.ascontiguousarray(points+initial_shift,dtype=np.float32)
                forward,ok,error=cv2.calcOpticalFlowPyrLK(old_roi,new_roi,points,initial,
                    winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                if forward is not None:
                    backward,back_ok,back_error=cv2.calcOpticalFlowPyrLK(new_roi,old_roi,forward,points.copy(),
                        winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                    if backward is not None:
                        src=global_points.reshape(-1,2);dst=forward.reshape(-1,2)+offset
                        keep=(ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm(backward.reshape(-1,2)+offset-src,axis=1)<=1.5)
                        keep&=(error.ravel()<=18)&(back_error.ravel()<=18)
                        keep&=np.isfinite(dst).all(1)&(dst[:,0]>=0)&(dst[:,0]<w)&(dst[:,1]>=0)&(dst[:,1]<h)
                        if keep.sum()>=4:
                            matrix,mask=cv2.estimateAffinePartial2D(src[keep],dst[keep],method=cv2.RANSAC,
                                ransacReprojThreshold=2.,maxIters=100,confidence=.99,refineIters=5)
                            if matrix is not None and mask is not None:
                                inliers=int(mask.sum());scale=float(np.linalg.norm(matrix[0,:2]))
                                angle=abs(math.degrees(math.atan2(matrix[1,0],matrix[0,0])))
                                shift=float(np.linalg.norm(matrix[:,2]))
                                if inliers>=4 and .7<=scale<=1.4 and angle<=15 and shift<=.18*math.hypot(w,h):
                                    corners=np.asarray([[a,b,1],[c,b,1],[c,d,1],[a,d,1]],np.float32)@matrix.T
                                    candidate=[float(corners[:,0].min()),float(corners[:,1].min()),
                                               float(corners[:,0].max()),float(corners[:,1].max())]
                                    reason='forward_backward_LK_and_robust_affine'
            predicted.append(candidate)
            evidence.append(dict(reason=reason,inliers=inliers,previous_box=list(box),predicted_box=candidate))
        return predicted,evidence

    def update(self,image,frame,controls,selected):
        if not selected.get('selected_valid'):
            self.previous=None
            return selected
        digest=frame_identity(image,'',0,0,0)['image_identity']
        if frame.get('image_identity')!=digest:
            self.previous=None
            return dict(selected,selected_valid=False,selected_state='unknown',
                        simulation_association_valid=False,reason='continuity_current_image_mismatch')
        boxes=selected.get('member_boxes')
        if not boxes:
            boxes=[selected['box']] if selected.get('box') else []
        if not boxes:
            self.previous=None
            return dict(selected,selected_valid=False,selected_state='unknown',
                        simulation_association_valid=False,reason='no_current_member_boxes')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        context=(frame['camera_id'],controls.lane_id,controls.movement,selected.get('selection_origin'))
        previous=self.previous;evidence=[];continued=False
        if (previous and previous['context']==context and previous['gray'].shape==gray.shape
                and frame['frame_id']>previous['frame_id']
                and 0<frame['captured_ms']-previous['captured_ms']<=self.max_gap_ms
                and len(previous['boxes'])==len(boxes)):
            hits=np.asarray([[box_iou(a,b)>=.5 for b in previous['boxes']] for a in boxes])
            continued=bool(np.all(hits.sum(0)==1) and np.all(hits.sum(1)==1))
            if not continued:
                predicted,evidence=self.predict_boxes(previous,gray,boxes)
                hits=np.asarray([[box_iou(a,b)>=.5 for b in predicted] for a in boxes])
                continued=bool(np.all(hits.sum(0)==1) and np.all(hits.sum(1)==1))
        if not continued:self.serial+=1
        self.previous=dict(context=context,gray=gray,boxes=[list(b) for b in boxes],
                           frame_id=frame['frame_id'],captured_ms=frame['captured_ms'])
        return dict(selected,prior_policy_head_id=selected.get('head_id'),
            head_id=f"{frame['camera_id']}:current_group:{self.serial}",
            continuity_corroborated=continued,continuity_motion=evidence,
            continuity_maximum_gap_ms=self.max_gap_ms,
            continuity_scope='identity_only_all_lights_and_colors_from_current_frame')
