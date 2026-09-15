import pathlib, urllib.request, hashlib, json, datetime
root=pathlib.Path(__file__).resolve().parents[1]
commit='26d18048e1ca8ff2b02c7016b993de48ed0760f5'
files={'Datasets/ComCat/ComCat_catalog.csv':('ComCat_catalog.csv','dab35c46d2ff46e20e389077655042ca92fb115a'),'LICENSE':('LICENSE','d645bca077ac0f197c84805e573f97f3ecf50ba4')}
out=root/'data/raw/earthquakenpp_comcat';out.mkdir(parents=True,exist_ok=True)
for remote,(name,expected) in files.items():
 p=out/name;url=f'https://raw.githubusercontent.com/ss15859/EarthquakeNPP/{commit}/{remote}'
 if p.exists():
  data=p.read_bytes()
 else:
  with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
 blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 assert blob==expected,(name,blob,expected)
 if not p.exists():p.write_bytes(data)
 provenance={'source_url':url,'commit':commit,'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Pinned catalog acquisition and content verification'}
 q=p.with_name(p.name+'.provenance.json')
 if not q.exists():q.write_text(json.dumps(provenance,indent=2)+'\n')
 print(name,len(data),blob)
print((out/'LICENSE').read_text())
print((out/'ComCat_catalog.csv').read_text().splitlines()[:3])
