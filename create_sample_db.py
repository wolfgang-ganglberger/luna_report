import pandas as pd
# create the list file
list_path = './sample.lst'
df = pd.DataFrame(data={'sid': ['koges_0006c472-953f-4076-b5cf-20414e4a78f4'], 
	'edf':['./koges_0006c472-953f-4076-b5cf-20414e4a78f4.edf'], 
	'xml':['./koges_0006c472-953f-4076-b5cf-20414e4a78f4.xml']})
df = df[['sid', 'edf', 'xml']]
df.to_csv(list_path, sep='\t', index=False, header=False)
