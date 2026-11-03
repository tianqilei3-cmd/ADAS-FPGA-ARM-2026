"""Linux USB-UART V2 road-only M1. Unknown lamp/navigation/motion stay invalid."""
import os,termios,fcntl,time,select


def packet(command,payload):
    raw=bytes([0x55,command,len(payload),*payload]);return raw+bytes([sum(raw)&255])


def packets(frame_id,roads,width,height):
    flags=sum((1<<bit) for name,bit in [('vehicle',0),('person',1),('two_wheeler',2),('traffic_light',5)] if any(r['name']==name for r in roads))
    lo,hi=frame_id&255,(frame_id>>8)&255
    out=[packet(0x11,[2,0,lo,hi,1,flags,0,0,0,0])]
    classes={'traffic_light':0,'vehicle':1,'two_wheeler':2,'person':3}
    ranked=sorted(roads,key=lambda r:-r['confidence'])[:4]
    for slot,r in enumerate(ranked):
        a,b,c,d=r['box'];coords=[max(0,min(1920,round(a*1920/width))),max(0,min(1080,round(b*1080/height))),
            max(0,min(1920,round(c*1920/width))),max(0,min(1080,round(d*1080/height)))]
        if coords[2]<=coords[0] or coords[3]<=coords[1]:continue
        values=[]
        for v in coords:values.extend([v&255,(v>>8)&255])
        out.append(packet(0x12,[2,0,lo,hi,slot,0,classes[r['name']],max(0,min(255,round(r['confidence']*255))),*values]))
    return out


class Uart:
    def __init__(self,path):
        self.fd=os.open(path,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK);fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attr=termios.tcgetattr(self.fd);attr[0]=0;attr[1]=0;attr[2]=termios.CS8|termios.CREAD|termios.CLOCAL
        attr[3]=0;attr[4]=termios.B115200;attr[5]=termios.B115200;attr[6][termios.VMIN]=0;attr[6][termios.VTIME]=0
        termios.tcsetattr(self.fd,termios.TCSANOW,attr);termios.tcflush(self.fd,termios.TCIOFLUSH)

    def send(self,data):
        sent=0
        while sent<len(data):sent+=os.write(self.fd,data[sent:])
        termios.tcdrain(self.fd)
        # The FPGA decoder waits for its reply to finish before the next packet.
        time.sleep(.005)
        chunks=[]
        while select.select([self.fd],[],[],0)[0]:
            part=os.read(self.fd,4096)
            if not part:break
            chunks.append(part)
        return b''.join(chunks)

    def close(self):os.close(self.fd)
