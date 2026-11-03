"""Portable V3 context and identified action feedback. No hardware I/O."""
import math

CLASSES={'traffic_light':0,'vehicle':1,'two_wheeler':2,'person':3}


def packet(command,payload):
    raw=bytes([0x55,command,len(payload),*payload])
    return raw+bytes([sum(raw)&255])


def context_packet(frame_id,context):
    if type(frame_id) is not int or not 0<=frame_id<=65535:
        raise ValueError('16-bit host frame ID required')
    keys=['camera_id','scene','valid_flags','object_flags','risk_flags','light_state','navigation',
          'control_flags','movement','boundary','epoch','age_units','arm_action']
    values={key:context.get(key,0) for key in keys}
    if any(type(value) is not int or not 0<=value<=255 for value in values.values()):
        raise ValueError('Context fields must be bytes')
    if (values['camera_id']>3 or values['scene']>4 or values['valid_flags']&128
            or values['object_flags']&192 or values['risk_flags']&240
            or values['light_state']>4 or values['navigation']>3 or values['control_flags']&192
            or values['movement']>3 or values['boundary']>2 or values['arm_action']>3):
        raise ValueError('Context enum or reserved-bit violation')
    return packet(0x13,[3,values['camera_id'],frame_id&255,frame_id>>8,
        values['scene'],values['valid_flags'],values['object_flags'],values['risk_flags'],
        values['light_state'],values['navigation'],values['control_flags'],values['movement'],
        values['boundary'],values['epoch'],values['age_units'],values['arm_action']])


def packets(frame_id,roads,width,height,context,max_boxes=4):
    if not width>0 or not height>0:raise ValueError('Image dimensions required')
    if type(max_boxes) is not int or not 0<=max_boxes<=4:raise ValueError('Invalid box budget')
    flags=sum(1<<bit for name,bit in [('vehicle',0),('person',1),('two_wheeler',2),('traffic_light',5)]
              if any(r['name']==name for r in roads))
    context=dict(context,object_flags=flags)
    out=[context_packet(frame_id,context)]
    for slot,row in enumerate(sorted(roads,key=lambda r:(r['name'] not in ('person','two_wheeler'),-r['confidence']))[:max_boxes]):
        if row['name'] not in CLASSES:continue
        a,b,c,d=row['box'];confidence=row['confidence']
        if not all(math.isfinite(v) for v in [a,b,c,d,confidence]):continue
        xy=[max(0,min(1920,round(a*1920/width))),max(0,min(1080,round(b*1080/height))),
            max(0,min(1920,round(c*1920/width))),max(0,min(1080,round(d*1080/height)))]
        if xy[2]<=xy[0] or xy[3]<=xy[1]:continue
        coords=[]
        for value in xy:coords.extend([value&255,value>>8])
        out.append(packet(0x12,[2,context.get('camera_id',0),frame_id&255,frame_id>>8,
            slot,0,CLASSES[row['name']],max(0,min(255,round(confidence*255))),*coords]))
    return out


def query_wire():return bytes([0x55,0x04,0x59])


class ReplyStream:
    def __init__(self):self.buffer=bytearray();self.invalid_bytes=0

    def feed(self,data):
        self.buffer.extend(data);result=[]
        while self.buffer:
            if self.buffer[0]!=0x55:
                del self.buffer[0];self.invalid_bytes+=1;continue
            if len(self.buffer)<2:break
            command=self.buffer[1]
            if command==0x84:
                if len(self.buffer)<3:break
                if self.buffer[2]!=12:
                    del self.buffer[0];self.invalid_bytes+=1;continue
                size=16
            elif command==3:
                if len(self.buffer)<3:break
                if self.buffer[2] not in (1,2):
                    del self.buffer[0];self.invalid_bytes+=1;continue
                size=4+self.buffer[2]
            else:size=3
            if len(self.buffer)<size:break
            raw=bytes(self.buffer[:size])
            if sum(raw[:-1])&255!=raw[-1]:
                del self.buffer[0];self.invalid_bytes+=1;continue
            result.append(raw);del self.buffer[:size]
        return result


def decode_feedback(raw):
    if len(raw)!=16 or raw[:3]!=bytes([0x55,0x84,12]) or sum(raw[:-1])&255!=raw[-1]:
        raise ValueError('Invalid identified feedback packet')
    p=raw[3:-1]
    if (p[0]!=3 or p[1]>3 or p[5]>4 or p[6]>6 or p[7]>7 or p[8]&248 or p[9]>3):
        raise ValueError('Unsupported feedback fields')
    return dict(version=p[0],camera_id=p[1],frame_id=p[2]+256*p[3],epoch=p[4],scene=p[5],
                action=p[6],alarm_code=p[7],decision_valid=bool(p[8]&1),link_timeout=bool(p[8]&2),
                seen_extended=bool(p[8]&4),reject_reason=p[9],build_tag=p[10],capabilities=p[11])


def feedback_matches(feedback,wire):
    if len(wire)!=20 or wire[:3]!=bytes([0x55,0x13,16]):raise ValueError('Expected context packet')
    p=wire[3:-1]
    return (feedback['version']==3 and feedback['build_tag'] in (0x27,0x28,0x29) and feedback['capabilities']&31==31
            and feedback['camera_id']==p[1] and feedback['frame_id']==p[2]+256*p[3]
            and feedback['epoch']==p[13] and feedback['scene']==p[4]
            and feedback['reject_reason']==0 and not feedback['link_timeout'])
