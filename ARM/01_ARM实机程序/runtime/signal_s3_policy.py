"""Portable prior PC policy; 500ms context continuity, virtual lanes only.
No truth labels are imported. Image identity follows the ARM pixel+shape hash.
"""
import cv2
import math
import numpy as np
from light_roi_common import NAMES,classify_policy,crop_head
from arm_scene_a04 import frame_identity
def image_identity(image):return frame_identity(image,'',0,0,0)['image_identity']


def interior_lamp_evidence(crop):
    """Separate an enclosed luminous component from colored crop-edge scenery.

    This is local corroboration of a highly confident red classification, not
    a whole-image light detector. Comparable interior colors remain ambiguous.
    """
    hue, saturation, value = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    # Current daylight development threshold; dim/night inputs remain unknown.
    eligible = (saturation >= 80) & (value >= 160)
    masks = [eligible & ((hue <= 12) | (hue >= 165)),
             eligible & (hue >= 15) & (hue <= 34), eligible & (hue >= 35) & (hue <= 95)]
    scores = []
    for mask in masks:
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components = [s for s in stats[1:] if s[0] > 0 and s[1] > 0
                      and s[0]+s[2] < crop.shape[1]-1 and s[1]+s[3] < crop.shape[0]-1
                      and min(s[2],s[3]) >= 2 and .2 <= s[2]/s[3] <= 5
                      and s[4] >= max(4, crop.shape[0]*crop.shape[1]*.003)]
        scores.append(max((int(s[4]) for s in components), default=0))
    order = np.argsort(-np.asarray(scores));first,second = map(int,order[:2])
    state = first if scores[first] >= 4 and scores[first] >= 3*max(1,scores[second]) else None
    return dict(interior_component_areas=scores,state=state,minimum_value=160)


def inside_vehicle_body(lamp_box,vehicles,min_confidence=.50,inset=.02):
    if len(lamp_box)!=4 or not all(math.isfinite(x) for x in lamp_box):
        raise ValueError('Invalid lamp box')
    x1,y1,x2,y2=lamp_box
    if x2<=x1 or y2<=y1:raise ValueError('Invalid lamp dimensions')
    if not 0<=min_confidence<=1 or not 0<=inset<.5:raise ValueError('Invalid vehicle veto settings')
    for vehicle in vehicles:
        if vehicle.get('name')!='vehicle' or vehicle.get('confidence',0)<min_confidence:continue
        box=vehicle['box']
        if len(box)!=4 or not all(math.isfinite(x) for x in box):continue
        a,b,c,d=box;dx,dy=(c-a)*inset,(d-b)*inset
        if c>a and d>b and x1>=a+dx and x2<=c-dx and y1>=b+dy and y2<=d-dy:
            return True
    return False


def suppress_contained_heads(rows):
    accepted=list(rows);removed=[];suppressed=set()
    for i,large in enumerate(accepted):
        if i in suppressed:continue
        a=np.asarray(large['box'],float);area_a=float(np.prod(a[2:]-a[:2]))
        for j,small in enumerate(accepted):
            if i==j or j in suppressed or small['name']!=large['name']:continue
            if small.get('attributes',{}).get('lamp_type')!=large.get('attributes',{}).get('lamp_type'):continue
            if large['confidence']<small['confidence']:continue
            b=np.asarray(small['box'],float);area_b=float(np.prod(b[2:]-b[:2]))
            if area_b<=0 or not 1.25<=area_a/area_b<=4.5:continue
            # Allow small localization jitter around the enclosing housing.
            pad=(a[2:]-a[:2])*.05
            intersection=float(np.maximum(0,np.minimum(a[2:]+pad,b[2:])-np.maximum(a[:2]-pad,b[:2])).prod())
            if intersection/area_b>=.90:
                suppressed.add(j);removed.append(dict(small,rejection='contained_duplicate',retained_box=large['box']))
    return [r for i,r in enumerate(accepted) if i not in suppressed],removed


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0., a[2:] - a[:2]).prod() + np.maximum(0., b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def guarded_signal_rows(image, classified):
    """Recheck actual crop pixels; never recover green from an unknown decision."""
    rows = []
    for candidate in classified:
        row = dict(candidate)
        crop = crop_head(image, row['box'])
        p = np.asarray(row['probabilities'], float)
        local = interior_lamp_evidence(crop)
        emission = row['visible_emission_probability']
        if (p.shape != (4,) or not np.isfinite(p).all() or (p < 0).any()
                or (p > 1).any() or abs(float(p.sum()) - 1) > .001
                or not np.isfinite(emission) or not 0 <= emission <= 1):
            row.update(name='unknown', scene_reason='invalid_color_probability')
            rows.append(row)
            continue
        decision, reason = classify_policy(p, crop, threshold=.85, margin=.20)
        if decision == 4 and reason == 'color_conflict' and p[0] >= .95 and local['state'] == 0:
            decision, reason = 0, 'strong_red_with_enclosed_red_component'
        state = NAMES[decision] if decision in (0, 1, 2) else 'unknown'
        if state == 'green' and (row['visible_emission_probability'] < .5 or local['state'] != 2):
            state, reason = 'unknown', 'green_without_bright_enclosed_emission'
        row.update(name=state, scene_reason=reason, interior_evidence=local)
        rows.append(row)
    return rows


class AutomaticSignalSelector:
    """Visible arrow/group -> virtual requested movement; no horizontal ordering.

    Round lights and undirected/multiple candidates remain unknown. Session IDs
    continue only on a fresh unique IoU match; missing observations clear them.
    """
    def __init__(self):
        self.previous = None
        self.serial = 0

    def select(self, rows, frame, controls):
        result = dict(**frame, automatic_head_location=True, selected_valid=False,
                      selection_origin='automatic_detector_and_current_arrow',
                      selected_state='unknown', simulation_association_valid=False,
                      physical_lane_association_verified=False,
                      association_scope='experimental_visible_arrow_to_virtual_movement',
                      lane_id=controls.lane_id, movement=controls.movement,
                      reason='no_confirmed_vehicle_arrow', head_id=None)
        vehicles = [r for r in rows if r.get('attributes', {}).get('lamp_type') == 'vehicle_signal'
                    and r.get('confidence', 0) >= .25 and not r.get('inside_vehicle', False)]
        vehicles, duplicates = suppress_contained_heads(vehicles)
        matching = [r for r in vehicles if r['attributes'].get('control_direction') == controls.movement]
        result.update(vehicle_candidates=len(vehicles), matching_arrow_candidates=len(matching),
                      duplicate_count=len(duplicates))
        if not matching:
            self.previous = None
            result['reason'] = ('round_or_unknown_direction_requires_approach_association' if vehicles
                                else 'no_confirmed_vehicle_arrow')
            return result
        is_group = len(matching) > 1
        if is_group:
            boxes = np.asarray([r['box'] for r in matching], float)
            heights = boxes[:, 3] - boxes[:, 1]
            centers_y = (boxes[:, 3] + boxes[:, 1]) / 2
            distinct = all(box_iou(a, b) < .1 for i, a in enumerate(boxes) for b in boxes[i+1:])
            same_row = np.ptp(centers_y) <= .5 * float(heights.min())
            similar_scale = heights.min() >= .67 * heights.max()
            states = {r['name'] for r in matching}
            if not (2 <= len(matching) <= 4 and distinct and same_row and similar_scale):
                self.previous = None
                return dict(result, reason='multiple_matching_heads_without_single_row_group_evidence')
            if len(states) != 1 or not states <= {'red', 'yellow', 'green'}:
                self.previous = None
                return dict(result, reason='matching_arrow_group_has_conflicting_or_unknown_color')
        row = matching[0]
        # Other undirected vehicle heads could control this same movement.
        unknown_directions = [r for r in vehicles
                              if r['attributes'].get('control_direction') not in ('left', 'straight', 'right')]
        if unknown_directions:
            self.previous = None
            return dict(result, reason='other_undirected_vehicle_heads_make_relation_ambiguous')
        previous = self.previous
        boxes = [r['box'] for r in matching]
        same_members = False
        if previous and len(previous['member_boxes']) == len(boxes):
            overlap = np.asarray([[box_iou(a, b) for b in previous['member_boxes']] for a in boxes])
            hits = overlap >= .5
            same_members = bool(np.all(hits.sum(0) == 1) and np.all(hits.sum(1) == 1))
        if (not previous or previous['camera_id'] != frame['camera_id']
                or previous['movement'] != controls.movement or previous['lane_id'] != controls.lane_id
                or frame['frame_id'] <= previous['frame_id']
                or not 0 < frame['captured_ms'] - previous['captured_ms'] <= 500
                or not same_members):
            self.serial += 1
        head_id = f"{frame['camera_id']}:auto:{self.serial}"
        self.previous = dict(frame, member_boxes=boxes, movement=controls.movement, lane_id=controls.lane_id)
        joined = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                  max(b[2] for b in boxes), max(b[3] for b in boxes)]
        return dict(result, selected_state=row['name'], selected_valid=row['name'] in ('red', 'yellow', 'green'),
                    simulation_association_valid=True, head_id=head_id, box=joined,
                    member_boxes=boxes, group_member_count=len(boxes),
                    reason='unanimous_current_arrow_row_group_for_virtual_movement' if is_group else 'unique_current_arrow_matches_virtual_movement',
                    attributes=row['attributes'])


def round_emitter(image, box, color):
    result = dict(valid=False, reason='no_unique_filled_round_emitter')
    if color not in ('red', 'yellow', 'green'):
        return dict(result, reason='unknown_current_color')
    a, b, c, d = map(float, box)
    pad_x, pad_y = (c-a)*.08, (d-b)*.08
    left, top = max(0, int(np.floor(a-pad_x))), max(0, int(np.floor(b-pad_y)))
    right, bottom = min(image.shape[1], int(np.ceil(c+pad_x))), min(image.shape[0], int(np.ceil(d+pad_y)))
    crop = image[top:bottom, left:right]
    if min(crop.shape[:2], default=0) < 4:
        return result
    h, s, v = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    hue = {'red': (h <= 12) | (h >= 165), 'yellow': (h >= 15) & (h <= 34),
           'green': (h >= 35) & (h <= 95)}[color]
    mask = (hue & (s >= 80) & (v >= 160)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    accepted = []
    for index in range(1, count):
        x, y, w, height, area = map(int, stats[index])
        if (area < 12 or min(w, height) < 4 or not .65 <= w/height <= 1.55
                or x <= 0 or y <= 0 or x+w >= crop.shape[1] or y+height >= crop.shape[0]):
            continue
        component = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        (_, _), radius = cv2.minEnclosingCircle(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = float(4*np.pi*cv2.contourArea(contour)/max(1., perimeter**2))
        # Pixel fill, unlike contour fill, rejects hollow digital zeroes/rings.
        fill = float(area/max(1., np.pi*(radius+.5)**2))
        if circularity < .68 or fill < .70:
            continue
        accepted.append(dict(center=(centers[index]+[left, top]).tolist(), radius=float(radius),
                             area=area, circularity=circularity, disk_fill=fill))
    if len(accepted) != 1:
        return dict(result, qualifying_components=len(accepted))
    return dict(result, valid=True, reason='one_current_filled_round_color_component', **accepted[0])




def forward_anchor(image, boxes, objects):
    sh,sw=image.shape[:2]
    low_x,high_x,head_bottom=float(boxes[:,0].min()),float(boxes[:,2].max()),float(boxes[:,3].max())
    anchors=[obj for obj in objects if obj['name']=='vehicle' and obj['confidence']>=.5
        and (.35*sw<=(obj['box'][0]+obj['box'][2])/2<=.65*sw or obj['box'][0]<=.5*sw<=obj['box'][2])
        and obj['box'][3]>=.65*sh and (obj['box'][1]+obj['box'][3])/2>=head_bottom+.015*sh
        and min(obj['box'][2],high_x+.06*sw)-max(obj['box'][0],low_x-.06*sw)>=.25*min(obj['box'][2]-obj['box'][0],high_x-low_x)]
    return max(anchors,key=lambda obj:obj['box'][3]) if anchors else None


def conservative_red_group(image,rows,objects,frame,controls,baseline):
    """Same-current-image frontal red consensus can restrict, never authorize GO.

    Digit/round ambiguity is allowed only for this stop-only warning. Physical
    lane applicability stays unverified. Green still uses the stricter policy.
    """
    if baseline.get('selected_valid') or controls.movement!='straight':return baseline
    if frame.get('image_identity')!=image_identity(image):return baseline
    vehicles=[r for r in rows if r.get('attributes',{}).get('lamp_type')=='vehicle_signal'
        and r.get('confidence',0)>=.25 and not r.get('inside_vehicle',False)]
    vehicles,_=suppress_contained_heads(vehicles)
    if not 2<=len(vehicles)<=4:return baseline
    for row in vehicles:
        p=row.get('probabilities',[0]*4)
        if (row['name']!='red' or len(p)!=4 or p[0]<.95
            or row.get('visible_emission_probability',0)<.5
            or interior_lamp_evidence(crop_head(image,row['box']))['state']!=0):
            return baseline
    boxes=np.asarray([r['box'] for r in vehicles],float)
    heights=boxes[:,3]-boxes[:,1];cy=(boxes[:,3]+boxes[:,1])/2
    if (np.ptp(cy)>.5*heights.min() or heights.min()<.67*heights.max()
        or any(not 1.3<=(b[2]-b[0])/(b[3]-b[1])<=4.5 for b in boxes)
        or any(box_iou(a,b)>=.1 for i,a in enumerate(boxes) for b in boxes[i+1:])):
        return baseline
    anchor=forward_anchor(image,boxes,objects)
    if anchor is None:return baseline
    return dict(baseline,selected_state='red',selected_valid=True,simulation_association_valid=True,
        physical_lane_association_verified=False,association_scope='stop_only_frontal_red_consensus_virtual_straight',
        selection_origin='current_red_vehicle_head_group_stop_only',reason='current_red_consensus_stop_only',
        stop_only=True,lane_id=controls.lane_id,movement='straight',
        head_id=f"{frame['camera_id']}:red-stop:{frame['frame_id']}",
        box=[float(boxes[:,0].min()),float(boxes[:,1].min()),float(boxes[:,2].max()),float(boxes[:,3].max())],
        member_boxes=boxes.tolist(),group_member_count=len(boxes),forward_traffic_anchor=anchor)


class RoundGroupSupplement:
    def __init__(self):
        self.previous = None
        self.serial = 0

    def refine(self, image, rows, objects, frame, controls, baseline):
        if baseline.get('selected_valid'):
            self.previous = None
            return dict(baseline, round_supplement_used=False)
        result = dict(baseline, round_supplement_used=False, round_evidence=[],
                      round_reason='round_supplement_not_applicable')

        def reject(reason):
            self.previous = None
            return dict(result, round_reason=reason)

        if frame.get('image_identity') != image_identity(image):
            return reject('round_current_image_mismatch')
        if controls.movement != 'straight':
            return reject('round_hypothesis_covers_virtual_straight_only')
        vehicles = [r for r in rows if r.get('attributes', {}).get('lamp_type') == 'vehicle_signal'
                    and r.get('confidence', 0) >= .25 and not r.get('inside_vehicle', False)]
        if any(r['attributes'].get('control_direction') == 'straight' for r in vehicles):
            return reject('do_not_override_ambiguous_or_unknown_straight_arrow')
        undecided = [r for r in vehicles if r['attributes'].get('control_direction') not in ('left', 'right')]
        proposals = []
        for index, row in enumerate(undecided):
            attr = row['attributes']
            pd = np.asarray(attr.get('direction_probabilities', []), float)
            shape_supported = (attr.get('shape') == 'round' or
                               (pd.shape == (4,) and np.isfinite(pd).all() and pd[0] >= .98))
            evidence = round_emitter(image, row['box'], row['name'])
            result['round_evidence'].append(dict(box=row['box'], color=row['name'],
                                                 shape_supported=bool(shape_supported), **evidence))
            if shape_supported and evidence['valid'] and row.get('visible_emission_probability', 0) >= .5:
                proposals.append(dict(row=row, index=index, evidence=evidence))
        groups = []
        for proposal in sorted(proposals, key=lambda p: -p['row']['confidence']):
            e = proposal['evidence']
            group = next((g for g in groups if g[0]['row']['name'] == proposal['row']['name']
                          and np.linalg.norm(np.asarray(g[0]['evidence']['center'])-e['center'])
                          <= max(2., .35*min(g[0]['evidence']['radius'], e['radius']))), None)
            if group is None:
                groups.append([proposal])
            else:
                group.append(proposal)
        members, assigned = [], set()
        for group in groups:
            housings = [p for p in group if 1.3 <= (p['row']['box'][2]-p['row']['box'][0]) /
                        (p['row']['box'][3]-p['row']['box'][1]) <= 4.5]
            if not housings:
                continue
            members.append(housings[0])
            assigned.update(p['index'] for p in group)
        if not 2 <= len(members) <= 4:
            return reject('requires_two_to_four_distinct_round_housings')
        boxes = np.asarray([p['row']['box'] for p in members], float)
        heights = np.asarray([2*p['evidence']['radius'] for p in members],float)
        cy = np.asarray([p['evidence']['center'][1] for p in members],float)
        if (np.ptp(cy) > .5*heights.min() or heights.min() < .67*heights.max()
                or any(box_iou(a, b) >= .1 for i, a in enumerate(boxes) for b in boxes[i+1:])):
            return reject('round_heads_not_one_supported_frontal_row')
        if len({p['row']['name'] for p in members}) != 1:
            return reject('round_group_current_color_conflict')
        for index, other in enumerate(undecided):
            if index in assigned:
                continue
            a = np.asarray(other['box'], float)
            # A nested digit/crop is not a second physical head. It cannot supply
            # a round member, but must not veto a separately evidenced emitter.
            contained = any(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
                            >= .9*np.prod(a[2:]-a[:2]) for b in boxes)
            if not contained or other['name'] not in (members[0]['row']['name'], 'unknown'):
                return reject('unresolved_other_vehicle_head')
        sh, sw = image.shape[:2]
        low_x, high_x, head_bottom = float(boxes[:, 0].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())
        anchors = [obj for obj in objects if obj['name'] == 'vehicle' and obj['confidence'] >= .5
                   and (.35*sw <= (obj['box'][0]+obj['box'][2])/2 <= .65*sw or obj['box'][0] <= .5*sw <= obj['box'][2])
                   and obj['box'][3] >= .65*sh and (obj['box'][1]+obj['box'][3])/2 >= head_bottom+.015*sh
                   and min(obj['box'][2],high_x+.06*sw)-max(obj['box'][0],low_x-.06*sw) >= .25*min(obj['box'][2]-obj['box'][0],high_x-low_x)]
        if not anchors:
            return reject('no_current_forward_traffic_context_for_round_hypothesis')
        anchor = max(anchors, key=lambda obj: obj['box'][3])
        previous = self.previous
        same_members = False
        if previous and len(previous['boxes']) == len(boxes):
            overlap = np.asarray([[box_iou(a, b) >= .5 for b in previous['boxes']] for a in boxes])
            same_members = bool(np.all(overlap.sum(0) == 1) and np.all(overlap.sum(1) == 1))
        if (not previous or previous['camera_id'] != frame['camera_id']
                or previous['lane_id'] != controls.lane_id or frame['frame_id'] <= previous['frame_id']
                or not 0 < frame['captured_ms']-previous['captured_ms'] <= 500 or not same_members):
            self.serial += 1
        self.previous = dict(frame, boxes=boxes.tolist(), lane_id=controls.lane_id)
        return dict(result, selected_state=members[0]['row']['name'], selected_valid=True,
                    simulation_association_valid=True, physical_lane_association_verified=False,
                    association_scope='experimental_round_group_and_forward_traffic_for_virtual_straight',
                    selection_origin='current_filled_round_emitters_frontal_group_and_traffic_context',
                    lane_id=controls.lane_id, movement='straight', head_id=f"{frame['camera_id']}:round_group:{self.serial}",
                    box=[low_x, float(boxes[:, 1].min()), high_x, head_bottom], member_boxes=boxes.tolist(),
                    group_member_count=len(boxes), round_supplement_used=True,
                    forward_traffic_anchor=anchor, round_reason='current_round_group_virtual_straight_hypothesis',
                    reason='current_round_group_virtual_straight_hypothesis')
