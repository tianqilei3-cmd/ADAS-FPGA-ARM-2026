"""Shared bounded downloads and JSON records for expanded relation data."""
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/signal_lane_relations_expanded_20261021'
AUTHOR = 'https://github.com/OpenDriveLab/OpenLane-V2/blob/master/data/README.md'
ANNOTATION_REVISION = '6c02353d42c777e2aab5a1056fbc80d1cc8fd798'
ANNOTATION_MD5 = '1c1f9d49ecd47d6bc5bf093f38fb68c9'
ANNOTATION_SHA256 = '902754cade6bbdc166d197770af81313d3ad610ec71bb8f1fd5951fda77a879c'
ANNOTATION_BYTES = 252265879
ANNOTATION_URL = ('https://huggingface.co/datasets/AlayaNeW/OpenDriveLab___OpenLane-V2/'
                  f'resolve/{ANNOTATION_REVISION}/raw/OpenLane-V2_subset_A_info-ls.tar')
IMAGE_REPO = 'Vishwa-307/OpenLaneV2'
IMAGE_REVISION = 'c282b8f7649b81f6e3e37ef61e30f31c480a0e4c'


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,algorithm).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def safe_relative(name):
    if not isinstance(name,str) or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Unsupported archive/image path')
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Path outside dataset')
    return path


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read_public(url,limit=2*1024**2):
    with opener().open(urllib.request.Request(url,headers={'User-Agent':'lane-relation-data/1.0'}),timeout=30) as r:
        data=r.read(limit+1)
        if len(data)>limit:
            raise ValueError('Metadata response exceeds limit')
        return data,dict(r.headers)


def download(url,target,expected_bytes,expected_sha256=None,expected_md5=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size!=expected_bytes:
            raise ValueError('Existing download size changed: '+str(target))
        if expected_sha256 and digest(target)!=expected_sha256:
            raise ValueError('Existing download SHA256 changed: '+str(target))
        if expected_md5 and digest(target,'md5')!=expected_md5:
            raise ValueError('Existing download MD5 changed: '+str(target))
        return
    partial=target.with_suffix(target.suffix+'.part')
    for attempt in range(1,5):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==expected_bytes:
            break
        headers={'User-Agent':'lane-relation-data/1.0'}
        if offset:
            headers['Range']=f'bytes={offset}-'
        try:
            with opener().open(urllib.request.Request(url,headers=headers),timeout=30) as response:
                if 'text/html' in response.headers.get('Content-Type',''):
                    raise ValueError('Download returned HTML')
                if response.status==206:
                    value=response.headers.get('Content-Range','')
                    if not value.startswith(f'bytes {offset}-') or not value.endswith('/'+str(expected_bytes)):
                        raise ValueError('Resume range/total size not confirmed')
                elif response.status==200:
                    offset=0
                else:
                    raise ValueError('Unexpected response status')
                announced=int(response.headers.get('Content-Length',0))
                if announced and offset+announced>expected_bytes:
                    raise ValueError('Download larger than pinned file')
                received=offset;notice=received+16*1024**2
                with partial.open('ab' if offset else 'wb') as stream:
                    while block:=response.read(1024**2):
                        received+=len(block)
                        if received>expected_bytes:
                            raise ValueError('File size limit exceeded')
                        stream.write(block)
                        if received>=notice:
                            print('RELATION_DATA_DOWNLOAD_MB',target.name,round(received/1024**2,1),flush=True)
                            notice+=16*1024**2
                if received!=expected_bytes:
                    raise OSError('Truncated public file; partial bytes retained')
            break
        except (OSError,TimeoutError) as error:
            print('RELATION_DOWNLOAD_RETRY',attempt,target.name,type(error).__name__,flush=True)
            if attempt==4:
                raise
    if partial.stat().st_size!=expected_bytes:
        raise ValueError('Incomplete downloaded file')
    if expected_sha256 and digest(partial)!=expected_sha256:
        raise ValueError('Downloaded SHA256 mismatch')
    if expected_md5 and digest(partial,'md5')!=expected_md5:
        raise ValueError('Downloaded MD5 mismatch')
    partial.replace(target)


def acquire_annotations():
    archive=OUT/'public_source/OpenLane-V2_subset_A_info-ls.tar'
    download(ANNOTATION_URL,archive,ANNOTATION_BYTES,ANNOTATION_SHA256,ANNOTATION_MD5)
    record=dict(created=datetime.now().isoformat(),author_source=AUTHOR,url=ANNOTATION_URL,
                archive=str(archive),bytes=ANNOTATION_BYTES,md5=ANNOTATION_MD5,sha256=ANNOTATION_SHA256,
                whole_author_archive_verified=True,pickle_loaded=False,hardware_accessed=False)
    output=OUT/'annotation_download.json'
    if not output.exists():
        save(output,record)
    print('FULL_RELATION_ANNOTATIONS_VERIFIED',str(archive),flush=True)
    return archive
