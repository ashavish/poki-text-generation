
import pandas as pd

poki_poems = pd.read_csv("poki.csv")

f = open("poki.txt","w")

for each in poki_poems.iterrows():
	title = each[1][1]
	poem = each[1][4]
	f.write(str(title))
	f.write(" : ")
	f.write(str(poem))
	f.write("\n")

f.close()