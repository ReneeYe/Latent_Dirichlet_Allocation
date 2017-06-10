import re

with open("ap.txt", "r") as f:
	temp = re.findall('<TEXT>(.*?)</TEXT>', f.read())
