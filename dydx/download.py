
URL = "https://raw.githubusercontent.com/Athpr123/Binary-Classification-Using-Machine-learning/refs/heads/master/dataset.csv"
import csv
import requests
import re 
from pprint import pprint
import pickle
from pathlib import Path 

res = requests.get(URL, allow_redirects=True)

data= res.content.decode().split('\r\n')

# split on commas except from inside double quotes
pattern = ",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"
def record(cols, entry):
	return {k:v for (k,v) in zip(cols, re.split(pattern, entry))}
	
cols = data[0].split(',')
# last element empty record ignore it 
data = data[1:]
records = [record(cols, row) for row in data ][:-1]

#ID,Age,Agency,Agency Type,Commision (in value),Destination,Distribution Channel,Duration,Gender,Net Sales,Product Name,Claim
	
	
agency_enc = { k:v for (v, k) in enumerate(set( [r['Agency'] for r in  records]) )}

agency_type_enc = { k:v for (v, k) in enumerate(set( [r['Agency Type'] for r in  records]))}

destination_enc =  {k:v for (v, k) in enumerate(set( [r['Destination'] for r in  records]))}

distribution_channel_enc = { k:v for (v, k) in enumerate( set( [r['Distribution Channel'] for r in  records])) }


gender_enc =  { k:v for (v, k) in enumerate(set( [r['Gender'] for r in  records]))}

product_name_enc = {k:v for (v, k) in enumerate(  set( [r['Product Name'] for r in  records]))}

encodings = {
'Agency': agency_enc ,
'Agency Type' : agency_type_enc,
'Destination' : destination_enc,
'Distribution Channel':distribution_channel_enc,
'Gender': gender_enc,
'Product Name': product_name_enc
}


encs = {'ID' : lambda x : int(x), 
'Age': lambda x: float(x),
'Agency': lambda x:agency_enc[x] ,
'Agency Type' : lambda x: agency_type_enc[x],
'Commision (in value)': lambda x: float(x),
'Destination' : lambda x: destination_enc[x],
'Distribution Channel': lambda x:distribution_channel_enc[x],
'Gender': lambda x : gender_enc[x],
'Net Sales': lambda x: float(x),
'Duration': lambda x: float(x),
'Product Name': lambda x : product_name_enc[x],
'Claim': lambda x: float(x),
}
assert len(cols) == len(encs), f'cols dont match encodings, {len(cols)} and {len(encs)} respectively '


records_encd = []
pairs = []
for (i,r) in enumerate(records):
	try:
		rec = {k:encs[k](v) for (k,v) in r.items()}
		records_encd.append(rec)
		pairs.append((records[i],rec))
	except ValueError:
		print(data[i])
		pass
	
# [ {k:encs[k](v) for (k,v) in r.items()} for r in records[:10]]

print(f'No records before encoding: {len(records)}')
print(f'No records after encoding: {len(records_encd)}')
pprint(records_encd[:2])

files= {'records.list':records, 'encoder.dict':encodings, 'encoded_records.list': records_encd}

p = Path(__file__).parent / 'data'
p.mkdir(parents=True, exist_ok=True)
print(p)

for (f,obj) in files.items():
	with open( p / f, 'wb') as fh:
		pickle.dump(obj, fh)
		
# testing saved data 

for f in files.keys():
	with open( p / f, 'rb') as fh:
		assert files[f] ==  pickle.load(fh), 'pickled file did not return same object' 


		






